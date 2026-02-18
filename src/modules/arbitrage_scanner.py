import asyncio
import json
from datetime import datetime, timezone
from src.core.config import Config
from src.core.polymarket_client import PolymarketClient
from src.utils.logger import setup_logger

log = setup_logger("arbitrage")


class ArbitrageScanner:
    """
    Single-market arbitrage scanner.
    Finds markets where YES + NO prices sum to less than $1.00.
    Buying both sides guarantees profit when the market resolves.

    Example: YES = $0.45, NO = $0.50 -> total = $0.95
    Buy both for $0.95, one side pays $1.00 -> guaranteed $0.05 profit (5.3% ROI).

    This happens due to liquidity gaps, especially on lower-volume markets.
    """

    def __init__(self, client: PolymarketClient):
        self.client = client
        self.found_opportunities: list[dict] = []

    async def scan_for_signals(self, max_markets: int = 50) -> list[dict]:
        """
        Scan top markets for YES+NO < $1.00 arbitrage opportunities.
        Returns signals for the profitable side (or both sides as separate signals).
        """
        signals = []
        min_edge = Config.ARB_MIN_EDGE  # minimum profit margin (e.g. 0.02 = 2%)

        log.info(f"Arbitrage scanner: checking top {max_markets} markets for pricing gaps...")

        markets = await self.client.get_markets(limit=max_markets, active=True)

        for market in markets:
            try:
                # Parse outcome prices
                outcomes_raw = market.get("outcomes", "[]")
                outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
                prices_raw = market.get("outcomePrices", "[]")
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else (prices_raw or [])
                clob_ids_raw = market.get("clobTokenIds", "[]")
                clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else (clob_ids_raw or [])

                if len(outcomes) != 2 or len(prices) != 2:
                    continue  # only binary markets (YES/NO)

                yes_price = float(prices[0])
                no_price = float(prices[1])
                total = yes_price + no_price

                if total <= 0 or yes_price <= 0 or no_price <= 0:
                    continue

                # Arbitrage exists when total < 1.00
                if total < (1.0 - min_edge):
                    profit_pct = ((1.0 / total) - 1) * 100
                    edge = 1.0 - total

                    condition_id = market.get("conditionId", "")
                    question = market.get("question", "")
                    liquidity = float(market.get("liquidityNum", 0))
                    volume_24h = float(market.get("volume24hr", 0))

                    # Skip very low liquidity (can't actually fill the trade)
                    if liquidity < Config.ARB_MIN_LIQUIDITY:
                        continue

                    # Buy the cheaper side as primary signal
                    if yes_price <= no_price:
                        buy_outcome = outcomes[0]
                        buy_price = yes_price
                        buy_token_id = clob_ids[0] if clob_ids else ""
                    else:
                        buy_outcome = outcomes[1]
                        buy_price = no_price
                        buy_token_id = clob_ids[1] if len(clob_ids) > 1 else ""

                    signal = {
                        "source": "arbitrage_scanner",
                        "type": "single_market_arb",
                        "condition_id": condition_id,
                        "market_question": question,
                        "recommended_outcome": buy_outcome,
                        "outcome": buy_outcome,
                        "token_id": buy_token_id,
                        "market_price": buy_price,
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "total_cost": total,
                        "guaranteed_profit_pct": round(profit_pct, 2),
                        "edge": round(edge, 4),
                        "avg_edge": round(edge, 4),
                        "liquidity": liquidity,
                        "volume_24h": volume_24h,
                        "confidence": min(0.95, 0.80 + edge * 2),  # high confidence for arb
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    signals.append(signal)
                    self.found_opportunities.append(signal)

                    log.info(
                        f"[ARB] {question[:60]} | "
                        f"YES=${yes_price:.4f} + NO=${no_price:.4f} = ${total:.4f} | "
                        f"Profit: {profit_pct:.1f}% | Liq: ${liquidity:,.0f}"
                    )

            except Exception as e:
                log.debug(f"Error checking market for arb: {e}")

            await asyncio.sleep(0.1)  # light rate limiting

        log.info(f"Arbitrage scan complete: {len(signals)} opportunities found")
        return signals

    async def scan_orderbook_arb(self, max_markets: int = 30) -> list[dict]:
        """
        Deeper scan: check actual orderbook prices (not just displayed prices).
        More accurate but slower due to individual orderbook fetches.
        """
        signals = []
        min_edge = Config.ARB_MIN_EDGE

        markets = await self.client.get_markets(limit=max_markets, active=True)

        for market in markets:
            try:
                outcomes_raw = market.get("outcomes", "[]")
                outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
                clob_ids_raw = market.get("clobTokenIds", "[]")
                clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else (clob_ids_raw or [])

                if len(outcomes) != 2 or len(clob_ids) != 2:
                    continue

                liquidity = float(market.get("liquidityNum", 0))
                if liquidity < Config.ARB_MIN_LIQUIDITY:
                    continue

                # Fetch real orderbook prices
                prices = await self.client.get_prices(clob_ids)
                if len(prices) != 2:
                    continue

                yes_price = prices.get(clob_ids[0], 0)
                no_price = prices.get(clob_ids[1], 0)
                total = yes_price + no_price

                if total > 0 and total < (1.0 - min_edge) and yes_price > 0 and no_price > 0:
                    profit_pct = ((1.0 / total) - 1) * 100
                    edge = 1.0 - total
                    condition_id = market.get("conditionId", "")
                    question = market.get("question", "")

                    buy_idx = 0 if yes_price <= no_price else 1
                    signal = {
                        "source": "arbitrage_scanner",
                        "type": "orderbook_arb",
                        "condition_id": condition_id,
                        "market_question": question,
                        "recommended_outcome": outcomes[buy_idx],
                        "outcome": outcomes[buy_idx],
                        "token_id": clob_ids[buy_idx],
                        "market_price": prices[clob_ids[buy_idx]],
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "total_cost": total,
                        "guaranteed_profit_pct": round(profit_pct, 2),
                        "edge": round(edge, 4),
                        "avg_edge": round(edge, 4),
                        "liquidity": liquidity,
                        "confidence": min(0.95, 0.85 + edge * 2),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    signals.append(signal)

                    log.info(
                        f"[ARB-OB] {question[:50]} | "
                        f"${yes_price:.4f} + ${no_price:.4f} = ${total:.4f} | "
                        f"Profit: {profit_pct:.1f}%"
                    )

            except Exception as e:
                log.debug(f"Orderbook arb check error: {e}")

            await asyncio.sleep(0.3)

        return signals
