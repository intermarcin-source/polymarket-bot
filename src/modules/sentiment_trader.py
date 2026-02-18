import asyncio
import json
from datetime import datetime, timezone
from typing import Optional
import anthropic
import httpx
from src.core.config import Config
from src.core.polymarket_client import PolymarketClient
from src.utils.logger import setup_logger

log = setup_logger("sentiment")

SENTIMENT_SYSTEM_PROMPT = """You are an expert prediction market probability estimator. Your job is to estimate the TRUE probability of events, then compare your estimate to the current market price to find mispricings.

You have access to your training data, general knowledge, and reasoning ability. Consider:
1. Base rates and historical precedents
2. Current political/economic/social context
3. Known biases in prediction markets (favorite-longshot bias, recency bias)
4. Information asymmetry - what does the market NOT know?

For each market, provide your honest probability estimate. If you think the market is efficient (correctly priced), say so. Only flag opportunities where you have genuine conviction.

IMPORTANT: Be calibrated. Don't just be contrarian. If a market is at 70% and you also think it's ~70%, say so.

Respond ONLY with valid JSON:
{
    "estimated_probability": 0.XX,
    "market_price": 0.XX,
    "edge": 0.XX,
    "direction": "buy_yes" or "buy_no" or "no_trade",
    "confidence": 0.XX,
    "reasoning": "Brief explanation",
    "sentiment_signals": ["signal1", "signal2"],
    "time_horizon": "hours" or "days" or "weeks"
}"""


class SentimentTrader:
    """
    LLM-powered sentiment and probability trading.

    Compares AI probability estimates against market prices to find mispricings.
    Uses Claude for deep analysis and Grok for real-time sentiment (Twitter/X data).

    Strategy: If the model estimates 75% but market shows 60%, buy YES.
    Only trades when there's a significant gap between estimated and market probability.
    """

    def __init__(self, client: PolymarketClient):
        self.client = client
        self.anthropic = anthropic.AsyncAnthropic(api_key=Config.ANTHROPIC_API_KEY)
        self._xai_client: Optional[httpx.AsyncClient] = None
        self.analysis_cache: dict[str, dict] = {}  # avoid re-analyzing same market

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

    async def _estimate_probability_claude(self, summary: dict) -> Optional[dict]:
        """Get Claude's probability estimate for a market."""
        prompt = f"""Estimate the TRUE probability of this prediction market outcome:

Market: {summary['question']}
Description: {summary.get('description', 'N/A')[:500]}
Category: {summary.get('category', 'N/A')}
End Date: {summary.get('end_date', 'N/A')}
Current YES price: ${summary.get('yes_price', 0):.4f}
Current NO price: ${summary.get('no_price', 0):.4f}
24h Volume: ${summary.get('volume_24h', 0):,.0f}
Liquidity: ${summary.get('liquidity', 0):,.0f}

Today: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

What is your estimated probability that YES wins? Compare to the market price and identify any mispricing."""

        try:
            response = await self.anthropic.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=800,
                system=SENTIMENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
        except Exception as e:
            log.debug(f"Claude sentiment error: {e}")
            return None

    async def _estimate_probability_grok(self, summary: dict) -> Optional[dict]:
        """Get Grok's probability estimate with real-time sentiment."""
        if not Config.XAI_API_KEY:
            return None

        prompt = f"""Estimate the TRUE probability of this prediction market outcome.
Factor in CURRENT social media sentiment, trending news, and public opinion.

Market: {summary['question']}
Description: {summary.get('description', 'N/A')[:500]}
Category: {summary.get('category', 'N/A')}
End Date: {summary.get('end_date', 'N/A')}
Current YES price: ${summary.get('yes_price', 0):.4f}

Today: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

What is the true probability? Is the market mispriced based on current sentiment?"""

        try:
            client = await self._get_xai_client()
            response = await client.post(
                "/chat/completions",
                json={
                    "model": "grok-3",
                    "messages": [
                        {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 800,
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
        except Exception as e:
            log.debug(f"Grok sentiment error: {e}")
            return None

    async def analyze_market(self, market: dict) -> Optional[dict]:
        """Analyze a market for sentiment-based mispricing."""
        condition_id = market.get("conditionId", "")
        question = market.get("question", "")

        # Skip if recently analyzed
        if condition_id in self.analysis_cache:
            cached = self.analysis_cache[condition_id]
            cache_age = (datetime.now(timezone.utc) -
                         datetime.fromisoformat(cached["timestamp"])).total_seconds()
            if cache_age < 1800:  # 30 minute cache
                return cached.get("signal")

        # Build summary
        outcomes = market.get("outcomes", [])
        prices_raw = market.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else (prices_raw or [])
        clob_ids_raw = market.get("clobTokenIds", "[]")
        clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else (clob_ids_raw or [])

        yes_price = float(prices[0]) if prices else 0
        no_price = float(prices[1]) if len(prices) > 1 else 0

        summary = {
            "question": question,
            "description": market.get("description", ""),
            "category": market.get("category", ""),
            "end_date": market.get("endDate", ""),
            "yes_price": yes_price,
            "no_price": no_price,
            "volume_24h": float(market.get("volume24hr", 0)),
            "liquidity": float(market.get("liquidityNum", 0)),
        }

        # Run both AIs in parallel
        claude_result, grok_result = await asyncio.gather(
            self._estimate_probability_claude(summary),
            self._estimate_probability_grok(summary),
            return_exceptions=True,
        )

        if isinstance(claude_result, Exception):
            claude_result = None
        if isinstance(grok_result, Exception):
            grok_result = None

        signal = self._build_signal(
            market, summary, condition_id, outcomes, clob_ids,
            yes_price, no_price, claude_result, grok_result
        )

        # Cache result
        self.analysis_cache[condition_id] = {
            "signal": signal,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return signal

    def _build_signal(
        self, market, summary, condition_id, outcomes, clob_ids,
        yes_price, no_price, claude_result, grok_result
    ) -> Optional[dict]:
        """Build a trading signal from AI probability estimates."""
        question = summary["question"]
        min_edge = Config.SENTIMENT_MIN_EDGE

        # Extract estimates
        claude_prob = float(claude_result.get("estimated_probability", 0)) if claude_result else None
        grok_prob = float(grok_result.get("estimated_probability", 0)) if grok_result else None
        claude_conf = float(claude_result.get("confidence", 0)) if claude_result else 0
        grok_conf = float(grok_result.get("confidence", 0)) if grok_result else 0

        # Need at least one valid estimate
        if claude_prob is None and grok_prob is None:
            return None

        # Weighted average (Claude 55%, Grok 45%) or single-model
        if claude_prob is not None and grok_prob is not None:
            est_prob = claude_prob * 0.55 + grok_prob * 0.45
            combined_conf = claude_conf * 0.55 + grok_conf * 0.45
        elif claude_prob is not None:
            est_prob = claude_prob
            combined_conf = claude_conf * 0.70  # discount for single model
        else:
            est_prob = grok_prob
            combined_conf = grok_conf * 0.65

        # Calculate edge
        yes_edge = est_prob - yes_price        # positive = underpriced YES
        no_edge = (1 - est_prob) - no_price    # positive = underpriced NO

        # Determine direction
        if yes_edge > min_edge and yes_edge >= no_edge:
            direction = "buy_yes"
            buy_outcome = outcomes[0] if outcomes else "Yes"
            buy_price = yes_price
            buy_token = clob_ids[0] if clob_ids else ""
            edge = yes_edge
        elif no_edge > min_edge:
            direction = "buy_no"
            buy_outcome = outcomes[1] if len(outcomes) > 1 else "No"
            buy_price = no_price
            buy_token = clob_ids[1] if len(clob_ids) > 1 else ""
            edge = no_edge
        else:
            return None  # market is fairly priced

        # Confidence threshold
        if combined_conf < Config.MIN_AI_CONFIDENCE:
            log.debug(f"Sentiment signal below confidence threshold: {combined_conf:.2f}")
            return None

        signal = {
            "source": "sentiment_trader",
            "type": "sentiment_mispricing",
            "condition_id": condition_id,
            "market_question": question,
            "recommended_outcome": buy_outcome,
            "outcome": buy_outcome,
            "token_id": buy_token,
            "market_price": buy_price,
            "estimated_probability": round(est_prob, 4),
            "claude_estimate": claude_prob,
            "grok_estimate": grok_prob,
            "edge": round(edge, 4),
            "avg_edge": round(edge, 4),
            "confidence": round(combined_conf, 3),
            "direction": direction,
            "reasoning": (
                (claude_result.get("reasoning", "") if claude_result else "") +
                " | " +
                (grok_result.get("reasoning", "") if grok_result else "")
            ).strip(" | "),
            "sentiment_signals": list(set(
                (claude_result.get("sentiment_signals", []) if claude_result else []) +
                (grok_result.get("sentiment_signals", []) if grok_result else [])
            )),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            f"[SENTIMENT] {question[:55]} | "
            f"Market: {buy_price:.2f} vs Est: {est_prob:.2f} | "
            f"Edge: {edge:+.2f} | Dir: {direction} | Conf: {combined_conf:.2f}"
        )
        return signal

    async def scan_for_signals(self, max_markets: int = 10) -> list[dict]:
        """Scan top markets for sentiment-based trading opportunities."""
        signals = []

        if not Config.ANTHROPIC_API_KEY:
            log.warning("No ANTHROPIC_API_KEY - sentiment trader disabled")
            return signals

        log.info(f"Sentiment trader scanning top {max_markets} markets...")

        markets = await self.client.get_markets(limit=max_markets)

        for market in markets:
            # Skip low-liquidity markets
            liquidity = float(market.get("liquidityNum", 0))
            if liquidity < 10000:
                continue

            try:
                signal = await self.analyze_market(market)
                if signal:
                    signals.append(signal)
            except Exception as e:
                log.error(f"Sentiment analysis error: {e}")

            await asyncio.sleep(1.5)  # rate limit AI calls

        log.info(f"Sentiment scan complete: {len(signals)} signals generated")
        return signals
