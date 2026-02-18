import json
import httpx
import asyncio
from typing import Optional
from src.core.config import Config
from src.utils.logger import setup_logger

log = setup_logger("polymarket")

DATA_API_URL = "https://data-api.polymarket.com"


class PolymarketClient:
    """Client for Polymarket public APIs (Gamma, CLOB, Data API)."""

    def __init__(self):
        self.clob_url = Config.CLOB_API_URL
        self.gamma_url = Config.GAMMA_API_URL
        self.data_url = DATA_API_URL
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Market Data ──────────────────────────────────────────

    async def get_markets(self, limit: int = 50, active: bool = True, closed: bool = False) -> list[dict]:
        """Fetch active prediction markets from Gamma API."""
        client = await self._get_client()
        params = {
            "limit": limit,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": "volume24hr",
            "ascending": "false",
        }
        try:
            resp = await client.get(f"{self.gamma_url}/markets", params=params)
            resp.raise_for_status()
            markets = resp.json()
            log.info(f"Fetched {len(markets)} markets")
            return markets
        except Exception as e:
            log.error(f"Failed to fetch markets: {e}")
            return []

    async def get_market(self, condition_id: str) -> Optional[dict]:
        """Fetch a single market by condition ID via CLOB API.

        The CLOB API supports path-based lookup: /markets/{condition_id}
        and returns tokens with token_id and outcome fields.
        """
        client = await self._get_client()
        try:
            resp = await client.get(f"{self.clob_url}/markets/{condition_id}")
            resp.raise_for_status()
            market = resp.json()
            if market and market.get("condition_id"):
                return market
            return None
        except Exception as e:
            log.debug(f"CLOB market lookup failed for {condition_id[:20]}...: {e}")
            # Fallback: try Gamma API with clobTokenIds
            try:
                resp = await client.get(
                    f"{self.gamma_url}/markets",
                    params={"limit": 100, "active": "true"}
                )
                resp.raise_for_status()
                for m in resp.json():
                    if m.get("conditionId") == condition_id:
                        # Convert Gamma format to CLOB-like format with tokens
                        outcomes = m.get("outcomes", [])
                        clob_ids_raw = m.get("clobTokenIds", "[]")
                        clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
                        tokens = []
                        for i, outcome in enumerate(outcomes):
                            tid = clob_ids[i] if i < len(clob_ids) else ""
                            tokens.append({"outcome": outcome, "token_id": tid})
                        m["tokens"] = tokens
                        return m
            except Exception as e2:
                log.debug(f"Gamma fallback also failed: {e2}")
            return None

    async def get_orderbook(self, token_id: str) -> Optional[dict]:
        """Fetch orderbook for a specific token (outcome)."""
        client = await self._get_client()
        try:
            resp = await client.get(f"{self.clob_url}/book", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"Failed to fetch orderbook for {token_id}: {e}")
            return None

    async def get_prices(self, token_ids: list[str]) -> dict[str, float]:
        """Get current mid-market prices for tokens."""
        prices = {}
        for token_id in token_ids:
            book = await self.get_orderbook(token_id)
            if book:
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                if bids and asks:
                    best_bid = float(bids[0]["price"])
                    best_ask = float(asks[0]["price"])
                    prices[token_id] = (best_bid + best_ask) / 2
                elif bids:
                    prices[token_id] = float(bids[0]["price"])
                elif asks:
                    prices[token_id] = float(asks[0]["price"])
        return prices

    # ── Events / Categories ──────────────────────────────────

    async def get_events(self, limit: int = 20) -> list[dict]:
        """Fetch events (groups of related markets)."""
        client = await self._get_client()
        try:
            params = {"limit": limit, "active": "true", "order": "volume24hr", "ascending": "false"}
            resp = await client.get(f"{self.gamma_url}/events", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"Failed to fetch events: {e}")
            return []

    # ── Trade Activity (Public Data API) ─────────────────────

    async def get_user_activity(self, address: str, limit: int = 100) -> list[dict]:
        """Fetch on-chain activity for a wallet via public Data API."""
        client = await self._get_client()
        try:
            params = {
                "user": address.lower(),
                "limit": min(limit, 500),
                "type": "TRADE",
                "sortBy": "TIMESTAMP",
                "sortDirection": "DESC",
            }
            resp = await client.get(f"{self.data_url}/activity", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"Failed to fetch activity for {address}: {e}")
            return []

    async def get_market_trades_events(self, token_id: str) -> list[dict]:
        """Fetch trade events for a market token (public, no auth)."""
        client = await self._get_client()
        try:
            resp = await client.get(
                f"{self.clob_url}/getMarketTradesEvents",
                params={"id": token_id},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"Failed to fetch trade events for {token_id}: {e}")
            return []

    # ── Leaderboard (Public Data API) ────────────────────────

    async def get_leaderboard(
        self,
        period: str = "ALL",
        order_by: str = "PNL",
        limit: int = 25,
    ) -> list[dict]:
        """Fetch the Polymarket leaderboard - top traders by PnL or volume."""
        client = await self._get_client()
        try:
            params = {
                "timePeriod": period,
                "orderBy": order_by,
                "limit": min(limit, 50),
            }
            resp = await client.get(f"{self.data_url}/v1/leaderboard", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"Failed to fetch leaderboard: {e}")
            return []

    # ── User Positions (Public Data API) ─────────────────────

    async def get_user_positions(self, address: str, limit: int = 100) -> list[dict]:
        """Fetch positions for a wallet address via public Data API."""
        client = await self._get_client()
        try:
            params = {
                "user": address.lower(),
                "limit": min(limit, 500),
                "sortBy": "CURRENT",
                "sortDirection": "DESC",
            }
            resp = await client.get(f"{self.data_url}/positions", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.debug(f"Data API positions failed for {address}, trying Gamma API")
            # Fallback to Gamma API
            try:
                params = {"user": address.lower()}
                resp = await client.get(f"{self.gamma_url}/positions", params=params)
                resp.raise_for_status()
                return resp.json()
            except Exception as e2:
                log.error(f"Failed to fetch positions for {address}: {e2}")
                return []

    # ── Market Summarization (for AI) ────────────────────────

    async def get_market_summary(self, market: dict) -> dict:
        """Create a clean summary of a market for AI analysis.

        Handles both CLOB API format (tokens array) and Gamma API format
        (outcomes + clobTokenIds + outcomePrices).
        """
        outcomes_list = []

        # CLOB API format: has "tokens" array with token_id + outcome
        tokens = market.get("tokens", [])
        if tokens:
            for t in tokens:
                outcomes_list.append({
                    "outcome": t.get("outcome", "Unknown"),
                    "token_id": t.get("token_id", ""),
                    "price": float(t.get("price", 0)),
                })
        else:
            # Gamma API format: outcomes + clobTokenIds + outcomePrices
            outcome_names = market.get("outcomes", [])
            clob_ids_raw = market.get("clobTokenIds", "[]")
            clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else (clob_ids_raw or [])
            prices_raw = market.get("outcomePrices", "[]")
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else (prices_raw or [])

            for i, name in enumerate(outcome_names):
                tid = clob_ids[i] if i < len(clob_ids) else ""
                price = float(prices[i]) if i < len(prices) else 0.0
                outcomes_list.append({
                    "outcome": name,
                    "token_id": str(tid),
                    "price": price,
                })

        return {
            "question": market.get("question", ""),
            "description": market.get("description", ""),
            "category": market.get("category", ""),
            "end_date": market.get("endDate", ""),
            "volume_24h": float(market.get("volume24hr", 0)),
            "total_volume": float(market.get("volumeNum", 0)),
            "liquidity": float(market.get("liquidityNum", 0)),
            "outcomes": outcomes_list,
            "condition_id": market.get("conditionId", market.get("condition_id", "")),
            "slug": market.get("slug", ""),
        }
