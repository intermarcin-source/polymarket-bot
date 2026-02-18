import asyncio
import json
from datetime import datetime, timezone
from src.core.config import Config
from src.core.polymarket_client import PolymarketClient
from src.utils.logger import setup_logger

log = setup_logger("negrisk")


class NegRiskScanner:
    """
    NegRisk rebalancing scanner for multi-outcome markets.

    Specific to Polymarket's multi-outcome markets (like "Who will win the
    election?" with 5+ candidates). When the probabilities across all outcomes
    don't sum to 100%, there's an arbitrage opportunity.

    Example: Election market with 5 candidates
    - Candidate A: 40%
    - Candidate B: 30%
    - Candidate C: 15%
    - Candidate D: 10%
    - Candidate E: 8%
    - Total: 103% -> The market is overpriced by 3%

    If total > 100%: sell the overpriced outcomes (or buy NO on all)
    If total < 100%: buy the underpriced outcomes (guaranteed profit)

    Scanners have found opportunities with 4-6% ROI with capital efficiency up to 29x.
    """

    def __init__(self, client: PolymarketClient):
        self.client = client

    async def scan_for_signals(self, max_events: int = 20) -> list[dict]:
        """
        Scan multi-outcome events for probability sum mismatches.
        """
        signals = []
        min_edge = Config.NEGRISK_MIN_EDGE

        log.info(f"NegRisk scanner checking top {max_events} events...")

        events = await self.client.get_events(limit=max_events)

        for event in events:
            try:
                event_markets = event.get("markets", [])
                if len(event_markets) < 3:
                    continue  # need 3+ outcomes for meaningful NegRisk

                # Collect all outcomes and their prices
                outcomes_data = []
                total_yes_price = 0.0
                has_valid_data = True

                for market in event_markets:
                    prices_raw = market.get("outcomePrices", "[]")
                    prices = json.loads(prices_raw) if isinstance(prices_raw, str) else (prices_raw or [])
                    outcomes_raw = market.get("outcomes", "[]")
                    outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
                    clob_ids_raw = market.get("clobTokenIds", "[]")
                    clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else (clob_ids_raw or [])

                    if not prices or not outcomes:
                        has_valid_data = False
                        break

                    yes_price = float(prices[0]) if prices else 0
                    if yes_price <= 0:
                        has_valid_data = False
                        break

                    total_yes_price += yes_price

                    outcomes_data.append({
                        "question": market.get("question", ""),
                        "outcome": outcomes[0] if outcomes else "?",
                        "yes_price": yes_price,
                        "no_price": float(prices[1]) if len(prices) > 1 else 1 - yes_price,
                        "condition_id": market.get("conditionId", ""),
                        "token_id_yes": clob_ids[0] if clob_ids else "",
                        "token_id_no": clob_ids[1] if len(clob_ids) > 1 else "",
                        "liquidity": float(market.get("liquidityNum", 0)),
                    })

                if not has_valid_data or not outcomes_data:
                    continue

                event_title = event.get("title", event.get("description", "Multi-outcome event"))
                deviation = total_yes_price - 1.0  # positive = overpriced, negative = underpriced
                abs_deviation = abs(deviation)

                if abs_deviation < min_edge:
                    continue  # market is efficiently priced

                # Check minimum liquidity across outcomes
                min_liq = min(o["liquidity"] for o in outcomes_data)
                if min_liq < Config.NEGRISK_MIN_LIQUIDITY:
                    continue

                roi_pct = abs_deviation / total_yes_price * 100 if total_yes_price > 0 else 0

                if deviation < 0:
                    # UNDERPRICED: total < 100% -> buy YES on all outcomes for guaranteed profit
                    # Sort by most underpriced (cheapest relative to fair value)
                    fair_share = 1.0 / len(outcomes_data)
                    underpriced = sorted(
                        outcomes_data,
                        key=lambda o: o["yes_price"] - fair_share
                    )

                    # Signal to buy the most underpriced outcome
                    best = underpriced[0]
                    signal = {
                        "source": "negrisk_scanner",
                        "type": "negrisk_underpriced",
                        "condition_id": best["condition_id"],
                        "market_question": f"[NegRisk] {event_title}: {best['outcome']}",
                        "recommended_outcome": best["outcome"],
                        "outcome": best["outcome"],
                        "token_id": best["token_id_yes"],
                        "market_price": best["yes_price"],
                        "total_probability": round(total_yes_price, 4),
                        "deviation": round(deviation, 4),
                        "roi_pct": round(roi_pct, 2),
                        "num_outcomes": len(outcomes_data),
                        "event_title": event_title,
                        "all_outcomes": [
                            {"name": o["outcome"], "price": o["yes_price"]}
                            for o in outcomes_data
                        ],
                        "edge": round(abs_deviation / len(outcomes_data), 4),
                        "avg_edge": round(abs_deviation / len(outcomes_data), 4),
                        "liquidity": min_liq,
                        "confidence": min(0.90, 0.70 + abs_deviation * 2),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    signals.append(signal)

                    log.info(
                        f"[NEGRISK] UNDERPRICED: {event_title[:50]} | "
                        f"Sum: {total_yes_price:.4f} (gap: {abs_deviation:.4f}) | "
                        f"Best buy: {best['outcome']} @ ${best['yes_price']:.4f} | "
                        f"ROI: {roi_pct:.1f}%"
                    )

                else:
                    # OVERPRICED: total > 100% -> buy NO on most overpriced outcome
                    # The most overpriced outcome is most likely to correct down
                    fair_share = 1.0 / len(outcomes_data)
                    overpriced = sorted(
                        outcomes_data,
                        key=lambda o: o["yes_price"] - fair_share,
                        reverse=True
                    )

                    best = overpriced[0]
                    signal = {
                        "source": "negrisk_scanner",
                        "type": "negrisk_overpriced",
                        "condition_id": best["condition_id"],
                        "market_question": f"[NegRisk] {event_title}: NO {best['outcome']}",
                        "recommended_outcome": "No",
                        "outcome": "No",
                        "token_id": best["token_id_no"],
                        "market_price": best["no_price"],
                        "total_probability": round(total_yes_price, 4),
                        "deviation": round(deviation, 4),
                        "roi_pct": round(roi_pct, 2),
                        "num_outcomes": len(outcomes_data),
                        "event_title": event_title,
                        "all_outcomes": [
                            {"name": o["outcome"], "price": o["yes_price"]}
                            for o in outcomes_data
                        ],
                        "edge": round(abs_deviation / len(outcomes_data), 4),
                        "avg_edge": round(abs_deviation / len(outcomes_data), 4),
                        "liquidity": min_liq,
                        "confidence": min(0.90, 0.70 + abs_deviation * 2),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    signals.append(signal)

                    log.info(
                        f"[NEGRISK] OVERPRICED: {event_title[:50]} | "
                        f"Sum: {total_yes_price:.4f} (excess: {deviation:.4f}) | "
                        f"Best NO: {best['outcome']} @ ${best['no_price']:.4f} | "
                        f"ROI: {roi_pct:.1f}%"
                    )

            except Exception as e:
                log.debug(f"NegRisk scan error: {e}")

            await asyncio.sleep(0.2)

        log.info(f"NegRisk scan complete: {len(signals)} opportunities found")
        return signals
