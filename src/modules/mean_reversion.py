import asyncio
import json
from datetime import datetime, timezone
from src.core.config import Config
from src.core.polymarket_client import PolymarketClient
from src.utils.logger import setup_logger

log = setup_logger("mean_reversion")


class MeanReversionTrader:
    """
    Mean reversion strategy for prediction markets.

    Detects when a market has crashed or spiked due to overreaction (rumors,
    panic, hype) and trades the expected bounce back toward fair value.

    Example: Market was at 60%, crashes to 20% on a rumor.
    If fundamentals haven't changed, buy at 20% expecting a reversion to ~40%.

    How it works:
    1. Track recent price history for markets
    2. Detect large price moves (>20% drop or spike)
    3. Assess if the move is an overreaction vs. legitimate information
    4. Trade the bounce with tight risk management

    Works because prediction markets often overreact to news, then correct.
    """

    def __init__(self, client: PolymarketClient):
        self.client = client
        self.price_history: dict[str, list[dict]] = {}  # condition_id -> price snapshots

    async def _record_prices(self, markets: list[dict]):
        """Record current prices to build history over time."""
        now = datetime.now(timezone.utc).isoformat()
        for market in markets:
            condition_id = market.get("conditionId", "")
            if not condition_id:
                continue

            prices_raw = market.get("outcomePrices", "[]")
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else (prices_raw or [])
            if not prices:
                continue

            yes_price = float(prices[0]) if prices else 0

            if condition_id not in self.price_history:
                self.price_history[condition_id] = []

            self.price_history[condition_id].append({
                "timestamp": now,
                "yes_price": yes_price,
                "volume_24h": float(market.get("volume24hr", 0)),
            })

            # Keep only last 50 snapshots per market
            if len(self.price_history[condition_id]) > 50:
                self.price_history[condition_id] = self.price_history[condition_id][-50:]

    def _detect_overreaction(self, condition_id: str, current_price: float) -> dict | None:
        """
        Detect if a market has had an overreaction.
        Returns info about the move if detected, None otherwise.
        """
        history = self.price_history.get(condition_id, [])
        if len(history) < 3:
            return None  # need at least 3 data points

        # Calculate recent price range
        recent_prices = [h["yes_price"] for h in history[-10:]]
        avg_recent = sum(recent_prices) / len(recent_prices)
        max_recent = max(recent_prices)
        min_recent = min(recent_prices)

        if avg_recent <= 0:
            return None

        # Large drop detection: current price well below recent average
        drop_from_avg = (avg_recent - current_price) / avg_recent
        if drop_from_avg > Config.MR_MIN_DROP:
            return {
                "type": "crash",
                "direction": "buy_yes",
                "avg_price": avg_recent,
                "max_price": max_recent,
                "current_price": current_price,
                "drop_pct": drop_from_avg * 100,
                "expected_reversion": (avg_recent + current_price) / 2,  # expect bounce to midpoint
            }

        # Large spike detection: current price well above recent average
        spike_from_avg = (current_price - avg_recent) / avg_recent
        if spike_from_avg > Config.MR_MIN_DROP:
            return {
                "type": "spike",
                "direction": "buy_no",
                "avg_price": avg_recent,
                "min_price": min_recent,
                "current_price": current_price,
                "spike_pct": spike_from_avg * 100,
                "expected_reversion": (avg_recent + current_price) / 2,
            }

        return None

    async def scan_for_signals(self, max_markets: int = 50) -> list[dict]:
        """
        Scan markets for mean reversion opportunities.
        First records current prices, then looks for overreactions.
        """
        signals = []

        log.info(f"Mean reversion trader scanning {max_markets} markets...")

        markets = await self.client.get_markets(limit=max_markets, active=True)

        # Record current prices (builds history over multiple cycles)
        await self._record_prices(markets)

        for market in markets:
            try:
                condition_id = market.get("conditionId", "")
                question = market.get("question", "")
                outcomes_raw = market.get("outcomes", "[]")
                outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
                prices_raw = market.get("outcomePrices", "[]")
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else (prices_raw or [])
                clob_ids_raw = market.get("clobTokenIds", "[]")
                clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else (clob_ids_raw or [])
                liquidity = float(market.get("liquidityNum", 0))
                volume_24h = float(market.get("volume24hr", 0))

                if not condition_id or len(prices) < 2 or len(outcomes) < 2:
                    continue

                if liquidity < Config.MR_MIN_LIQUIDITY:
                    continue

                yes_price = float(prices[0])
                no_price = float(prices[1])

                # Check for overreaction
                overreaction = self._detect_overreaction(condition_id, yes_price)
                if not overreaction:
                    continue

                # Determine buy side
                if overreaction["direction"] == "buy_yes":
                    buy_outcome = outcomes[0]
                    buy_price = yes_price
                    buy_token = clob_ids[0] if clob_ids else ""
                else:
                    buy_outcome = outcomes[1] if len(outcomes) > 1 else "No"
                    buy_price = no_price
                    buy_token = clob_ids[1] if len(clob_ids) > 1 else ""

                # Skip extreme prices
                if buy_price < 0.05 or buy_price > 0.95:
                    continue

                expected_reversion = overreaction["expected_reversion"]
                edge = abs(expected_reversion - buy_price)
                move_pct = overreaction.get("drop_pct", overreaction.get("spike_pct", 0))

                # Confidence scales with the size of the move and volume
                confidence = min(
                    0.85,
                    0.50
                    + min(move_pct / 100, 0.20)  # larger moves = more likely to revert
                    + min(volume_24h / 200000, 0.10)  # high volume = more data
                    + (0.05 if liquidity > 50000 else 0)
                )

                signal = {
                    "source": "mean_reversion",
                    "type": overreaction["type"] + "_reversion",
                    "condition_id": condition_id,
                    "market_question": question,
                    "recommended_outcome": buy_outcome,
                    "outcome": buy_outcome,
                    "token_id": buy_token,
                    "market_price": buy_price,
                    "avg_recent_price": round(overreaction["avg_price"], 4),
                    "expected_reversion_price": round(expected_reversion, 4),
                    "move_pct": round(move_pct, 2),
                    "move_type": overreaction["type"],
                    "edge": round(edge, 4),
                    "avg_edge": round(edge, 4),
                    "liquidity": liquidity,
                    "volume_24h": volume_24h,
                    "confidence": round(confidence, 3),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                signals.append(signal)

                log.info(
                    f"[MR] {overreaction['type'].upper()} detected: {question[:45]} | "
                    f"Avg: {overreaction['avg_price']:.2f} -> Now: {buy_price:.2f} "
                    f"({move_pct:+.1f}%) | "
                    f"Expected reversion: {expected_reversion:.2f}"
                )

            except Exception as e:
                log.debug(f"Mean reversion scan error: {e}")

        log.info(f"Mean reversion scan complete: {len(signals)} signals generated "
                 f"(tracking {len(self.price_history)} markets)")
        return signals
