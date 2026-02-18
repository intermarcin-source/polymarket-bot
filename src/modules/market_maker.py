import asyncio
import json
from datetime import datetime, timezone
from src.core.config import Config
from src.core.polymarket_client import PolymarketClient
from src.utils.logger import setup_logger

log = setup_logger("market_maker")


class MarketMaker:
    """
    Spread capture / market making strategy.

    Buy YES at bid, sell at ask, capturing the bid-ask spread.
    Example: Buy YES at $0.48, sell at $0.52 -> $0.04 spread profit per share.

    Works best on:
    - High-volume markets (frequent fills)
    - Markets with wide spreads (more profit per trade)
    - Stable markets (less directional risk while holding)

    In simulation mode, we identify markets with attractive spreads and
    generate signals to buy the cheaper side, expecting to capture the spread.
    """

    def __init__(self, client: PolymarketClient):
        self.client = client

    async def scan_for_signals(self, max_markets: int = 30) -> list[dict]:
        """
        Scan markets for wide bid-ask spreads worth capturing.
        Returns signals for markets where the spread provides good ROI.
        """
        signals = []
        min_spread = Config.MM_MIN_SPREAD  # minimum spread width (e.g. 0.03 = 3 cents)
        min_liquidity = Config.MM_MIN_LIQUIDITY

        log.info(f"Market maker scanning top {max_markets} markets for spreads...")

        markets = await self.client.get_markets(limit=max_markets, active=True)

        for market in markets:
            try:
                outcomes_raw = market.get("outcomes", "[]")
                outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
                clob_ids_raw = market.get("clobTokenIds", "[]")
                clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else (clob_ids_raw or [])
                liquidity = float(market.get("liquidityNum", 0))
                volume_24h = float(market.get("volume24hr", 0))
                condition_id = market.get("conditionId", "")
                question = market.get("question", "")

                if len(clob_ids) < 2 or liquidity < min_liquidity:
                    continue

                # Check orderbook for each outcome
                for i, token_id in enumerate(clob_ids[:2]):
                    book = await self.client.get_orderbook(token_id)
                    if not book:
                        continue

                    bids = book.get("bids", [])
                    asks = book.get("asks", [])

                    if not bids or not asks:
                        continue

                    best_bid = float(bids[0]["price"])
                    best_ask = float(asks[0]["price"])
                    spread = best_ask - best_bid

                    if spread < min_spread:
                        continue

                    # Check depth - need enough liquidity on both sides
                    bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
                    ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])

                    if bid_depth < 50 or ask_depth < 50:
                        continue

                    mid_price = (best_bid + best_ask) / 2
                    spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0

                    # ROI calculation: buy at bid, sell at ask
                    roi_pct = (spread / best_bid) * 100 if best_bid > 0 else 0

                    outcome_name = outcomes[i] if i < len(outcomes) else f"Outcome {i}"

                    signal = {
                        "source": "market_maker",
                        "type": "spread_capture",
                        "condition_id": condition_id,
                        "market_question": question,
                        "recommended_outcome": outcome_name,
                        "outcome": outcome_name,
                        "token_id": token_id,
                        "market_price": mid_price,  # use mid for slippage check
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread": round(spread, 4),
                        "spread_pct": round(spread_pct, 2),
                        "roi_pct": round(roi_pct, 2),
                        "mid_price": round(mid_price, 4),
                        "bid_depth": round(bid_depth, 2),
                        "ask_depth": round(ask_depth, 2),
                        "edge": round(spread / 2, 4),  # half-spread as edge
                        "avg_edge": round(spread / 2, 4),
                        "liquidity": liquidity,
                        "volume_24h": volume_24h,
                        "confidence": min(0.80, 0.55 + spread_pct * 0.05 + min(volume_24h / 100000, 0.15)),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    signals.append(signal)

                    log.info(
                        f"[MM] {question[:45]} | {outcome_name} | "
                        f"Bid: ${best_bid:.4f} Ask: ${best_ask:.4f} | "
                        f"Spread: ${spread:.4f} ({spread_pct:.1f}%) | "
                        f"ROI: {roi_pct:.1f}%"
                    )

                await asyncio.sleep(0.3)  # rate limit orderbook fetches

            except Exception as e:
                log.debug(f"Market maker scan error: {e}")

        # Sort by ROI potential
        signals.sort(key=lambda s: s.get("roi_pct", 0), reverse=True)
        signals = signals[:15]  # cap at top 15

        log.info(f"Market maker scan complete: {len(signals)} spread opportunities found")
        return signals
