import asyncio
import json
import httpx
from datetime import datetime, timezone
from typing import Optional
import anthropic
from src.core.config import Config
from src.core.polymarket_client import PolymarketClient
from src.utils.logger import setup_logger

log = setup_logger("ai_analyst")

SYSTEM_PROMPT = """You are an expert prediction market analyst and trader. You analyze Polymarket prediction markets to identify high-conviction trading opportunities.

Your job:
1. Assess the TRUE probability of each outcome based on fundamentals, news, logic, and data.
2. Compare your assessed probability to the current market price.
3. Identify mispricings where the market is significantly wrong.

Rules:
- Only recommend trades where you see at least a 10% edge (your probability vs market price).
- Consider liquidity, time to resolution, and information asymmetry.
- Be contrarian when evidence supports it, but don't be contrarian for its own sake.
- Provide clear reasoning for every recommendation.
- Rate your confidence from 0.0 to 1.0.

Respond ONLY with valid JSON in this exact format:
{
    "market_question": "...",
    "analysis": "Your 2-3 sentence analysis of the market",
    "recommended_outcome": "Yes" or "No" or null if no trade,
    "your_probability": 0.XX,
    "market_price": 0.XX,
    "edge": 0.XX,
    "confidence": 0.XX,
    "reasoning": "Why you believe the market is mispriced",
    "risk_factors": ["factor1", "factor2"],
    "time_sensitivity": "low" or "medium" or "high"
}

If you see no good trade, set recommended_outcome to null and confidence to 0."""


class AIAnalyst:
    """
    Dual-AI analysis engine using Claude Opus 4.6 and xAI SuperGrok.
    Markets are analyzed by both models independently, and trades are
    only recommended when both agree with high confidence.
    """

    def __init__(self, client: PolymarketClient):
        self.client = client
        self.anthropic = anthropic.AsyncAnthropic(api_key=Config.ANTHROPIC_API_KEY)
        self._xai_client: Optional[httpx.AsyncClient] = None

    async def _get_xai_client(self) -> httpx.AsyncClient:
        if self._xai_client is None or self._xai_client.is_closed:
            self._xai_client = httpx.AsyncClient(
                base_url="https://api.x.ai/v1",
                headers={
                    "Authorization": f"Bearer {Config.XAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._xai_client

    async def close(self):
        if self._xai_client and not self._xai_client.is_closed:
            await self._xai_client.aclose()

    async def _ask_claude(self, market_summary: dict) -> Optional[dict]:
        """Get Claude Opus 4.6's analysis of a market."""
        prompt = f"""Analyze this Polymarket prediction market and determine if there's a trading opportunity:

Market Question: {market_summary['question']}
Description: {market_summary.get('description', 'N/A')}
Category: {market_summary.get('category', 'N/A')}
End Date: {market_summary.get('end_date', 'N/A')}

Current Prices (probabilities):
{json.dumps(market_summary.get('outcomes', []), indent=2)}

24h Volume: ${market_summary.get('volume_24h', 0):,.0f}
Total Volume: ${market_summary.get('total_volume', 0):,.0f}
Liquidity: ${market_summary.get('liquidity', 0):,.0f}

Today's date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

Analyze this market and respond with your JSON assessment."""

        try:
            response = await self.anthropic.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            log.error(f"Claude returned invalid JSON: {e}")
            return None
        except Exception as e:
            log.error(f"Claude API error: {e}")
            return None

    async def _ask_grok(self, market_summary: dict) -> Optional[dict]:
        """Get SuperGrok's analysis of a market."""
        prompt = f"""Analyze this Polymarket prediction market and determine if there's a trading opportunity:

Market Question: {market_summary['question']}
Description: {market_summary.get('description', 'N/A')}
Category: {market_summary.get('category', 'N/A')}
End Date: {market_summary.get('end_date', 'N/A')}

Current Prices (probabilities):
{json.dumps(market_summary.get('outcomes', []), indent=2)}

24h Volume: ${market_summary.get('volume_24h', 0):,.0f}
Total Volume: ${market_summary.get('total_volume', 0):,.0f}
Liquidity: ${market_summary.get('liquidity', 0):,.0f}

Today's date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

Consider real-time social media sentiment, trending news, and public opinion.
Analyze this market and respond with your JSON assessment."""

        try:
            client = await self._get_xai_client()
            response = await client.post(
                "/chat/completions",
                json={
                    "model": "grok-3",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.3,
                },
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            log.error(f"Grok returned invalid JSON: {e}")
            return None
        except Exception as e:
            log.error(f"Grok API error: {e}")
            return None

    async def analyze_market(self, market: dict) -> Optional[dict]:
        """
        Analyze a single market with both AI models.
        Returns a combined signal only if both models agree.
        """
        summary = await self.client.get_market_summary(market)
        if not summary["question"]:
            return None

        log.info(f"Analyzing: {summary['question'][:80]}...")

        # Run both AIs in parallel
        claude_result, grok_result = await asyncio.gather(
            self._ask_claude(summary),
            self._ask_grok(summary),
            return_exceptions=True,
        )

        if isinstance(claude_result, Exception):
            log.error(f"Claude error: {claude_result}")
            claude_result = None
        if isinstance(grok_result, Exception):
            log.error(f"Grok error: {grok_result}")
            grok_result = None

        if not claude_result and not grok_result:
            log.warning(f"Both AIs failed for: {summary['question'][:60]}")
            return None

        # Check for agreement
        signal = self._combine_analyses(summary, claude_result, grok_result)
        return signal

    def _combine_analyses(
        self, summary: dict, claude: Optional[dict], grok: Optional[dict]
    ) -> Optional[dict]:
        """Combine both AI analyses into a single signal."""

        claude_rec = claude.get("recommended_outcome") if claude else None
        grok_rec = grok.get("recommended_outcome") if grok else None
        claude_conf = float(claude.get("confidence") or 0) if claude else 0
        grok_conf = float(grok.get("confidence") or 0) if grok else 0

        # Both must recommend a trade
        if not claude_rec and not grok_rec:
            return None

        # If both agree on the same outcome
        if claude_rec and grok_rec and claude_rec == grok_rec:
            combined_confidence = (claude_conf * 0.55 + grok_conf * 0.45)
            claude_edge = float(claude.get("edge") or 0) if claude else 0
            grok_edge = float(grok.get("edge") or 0) if grok else 0
            avg_edge = claude_edge * 0.55 + grok_edge * 0.45

            if combined_confidence < Config.MIN_AI_CONFIDENCE:
                log.info(
                    f"AIs agree on {claude_rec} but confidence too low: {combined_confidence:.2f}"
                )
                return None

            # Find the matching token
            token_id = ""
            market_price = 0.0
            for outcome in summary.get("outcomes", []):
                if outcome["outcome"].lower() == claude_rec.lower():
                    token_id = outcome["token_id"]
                    market_price = outcome["price"]
                    break

            signal = {
                "source": "ai_analyst",
                "type": "ai_consensus",
                "condition_id": summary["condition_id"],
                "market_question": summary["question"],
                "recommended_outcome": claude_rec,
                "token_id": token_id,
                "market_price": market_price,
                "claude_probability": float(claude.get("your_probability") or 0) if claude else 0,
                "grok_probability": float(grok.get("your_probability") or 0) if grok else 0,
                "avg_edge": avg_edge,
                "confidence": combined_confidence,
                "claude_reasoning": claude.get("reasoning", "") if claude else "",
                "grok_reasoning": grok.get("reasoning", "") if grok else "",
                "risk_factors": list(set(
                    (claude.get("risk_factors", []) if claude else [])
                    + (grok.get("risk_factors", []) if grok else [])
                )),
                "time_sensitivity": (
                    claude.get("time_sensitivity", "low") if claude else
                    grok.get("time_sensitivity", "low") if grok else "low"
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            log.info(
                f"[SIGNAL] AI CONSENSUS: {claude_rec} on '{summary['question'][:60]}' "
                f"(conf: {combined_confidence:.2f}, edge: {avg_edge:.2f})"
            )
            return signal

        # If only one AI has a recommendation with very high confidence
        if claude_rec and claude_conf >= 0.85 and not grok_rec:
            log.info(f"Claude-only signal (high conf): {claude_rec} ({claude_conf:.2f})")
            # Still return but with reduced confidence
            token_id = ""
            market_price = 0.0
            for outcome in summary.get("outcomes", []):
                if outcome["outcome"].lower() == claude_rec.lower():
                    token_id = outcome["token_id"]
                    market_price = outcome["price"]
                    break

            return {
                "source": "ai_analyst",
                "type": "ai_single_high_conviction",
                "model": "claude",
                "condition_id": summary["condition_id"],
                "market_question": summary["question"],
                "recommended_outcome": claude_rec,
                "token_id": token_id,
                "market_price": market_price,
                "confidence": claude_conf * 0.7,  # discount for single-model
                "avg_edge": float(claude.get("edge") or 0),
                "reasoning": claude.get("reasoning", ""),
                "risk_factors": claude.get("risk_factors", []),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Disagreement - no trade
        if claude_rec and grok_rec and claude_rec != grok_rec:
            log.info(
                f"AI DISAGREEMENT on '{summary['question'][:60]}': "
                f"Claude={claude_rec}({claude_conf:.2f}) vs Grok={grok_rec}({grok_conf:.2f})"
            )

        return None

    async def scan_for_signals(self, max_markets: int = 15) -> list[dict]:
        """
        Scan top markets and generate AI-based trade signals.
        """
        signals = []
        log.info(f"AI analyst scanning top {max_markets} markets...")

        markets = await self.client.get_markets(limit=max_markets)

        for market in markets:
            # Skip low-liquidity markets
            liquidity = float(market.get("liquidityNum", 0))
            if liquidity < 5000:
                continue

            try:
                signal = await self.analyze_market(market)
                if signal:
                    signals.append(signal)
            except Exception as e:
                log.error(f"Error analyzing market '{market.get('question', '?')[:40]}': {e}")

            await asyncio.sleep(1)  # rate limiting between AI calls

        log.info(f"AI analysis complete: {len(signals)} signals generated")
        return signals
