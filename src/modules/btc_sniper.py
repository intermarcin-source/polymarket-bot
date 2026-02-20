"""
BTC 5-Minute Sniper — Multi-Indicator Strategy
================================================
Trades Polymarket BTC Up/Down 5-minute markets using a composite
signal from CVD (order flow), large trade detection, and price delta.

Enters at T-240s (1 min into window) when prices are $0.50-$0.55,
using dynamic price caps based on signal strength.

Resolution source: Chainlink BTC/USD oracle.
Rule: Up wins if end_price >= start_price (flat = Up).
"""

import asyncio
import json
import time
import math
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp
import websockets

from src.utils.logger import setup_logger

log = setup_logger("btc_sniper")

# Polymarket BTC 5-min market constants
WINDOW_SECONDS = 300  # 5 minutes
SLUG_PREFIX = "btc-updown-5m-"

# Polymarket RTDS for Chainlink prices (REST fallback — WS needs special auth)
RTDS_REST_URL = "https://data.chain.link/streams/btc-usd"

# Binance WebSocket — aggregated trades (less noise than raw @trade)
BINANCE_WS_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@aggTrade"

# ── STRATEGY PARAMETERS ───────────────────────────────────────
ENTRY_SECONDS_BEFORE_CLOSE = 240  # Enter at T-240s (1 min into window)
SKIP_CUTOFF_SECONDS = 60          # Stop trying after T-60s (liquidity dries up)

# ── COMPOSITE SCORING ─────────────────────────────────────────
WEIGHT_CVD = 0.40            # CVD is strongest short-term predictor
WEIGHT_PRICE_DELTA = 0.30    # Price delta (what the market resolves on)
WEIGHT_LARGE_TRADES = 0.30   # Whale trades are very directional
MIN_SIGNAL_SCORE = 0.45      # Minimum composite score to generate signal
AGREEMENT_BONUS = 0.10       # Bonus when 2+ indicators agree on direction

# ── DYNAMIC PRICE CAP ─────────────────────────────────────────
# Stronger signal = accept higher entry prices (lower ROI but higher win rate)
# Format: (min_score_threshold, max_buy_price)
PRICE_TIERS = [
    (0.85, 0.75),  # Very strong signal → accept up to $0.75 (33% ROI)
    (0.70, 0.65),  # Strong signal → up to $0.65 (54% ROI)
    (0.55, 0.60),  # Medium signal → up to $0.60 (67% ROI)
    (0.45, 0.58),  # Weak signal → up to $0.58 (72% ROI, need 58% WR)
]

# ── CVD THRESHOLDS (USD volume) — Calibrated from overnight data ──
CVD_STRONG_THRESHOLD = 500_000     # $500K net = very strong (was $2M — never reached)
CVD_MEDIUM_THRESHOLD = 150_000     # $150K net = moderate (was $500K — rarely reached)
CVD_WEAK_THRESHOLD = 50_000        # $50K net = weak (was $100K — too conservative)

# ── LARGE TRADE DETECTION — Calibrated from overnight data ────
LARGE_TRADE_THRESHOLD = 100_000    # $100K = "large" trade (was $500K — never triggered)
WHALE_TRADE_THRESHOLD = 500_000    # $500K = whale trade (was $1M)
LARGE_TRADE_WINDOW_SECS = 60       # Look at large trades in last 60s


class BTCSniper:
    """
    Monitors BTC price and order flow via WebSocket streams and generates
    buy signals using a composite scoring system (CVD + delta + whale detection).
    Enters at T-240s when prices are $0.50-$0.55, with dynamic price caps
    based on signal strength.
    """

    def __init__(self, clob_client=None):
        self.clob_client = clob_client

        # Price tracking
        self.binance_price: float = 0.0
        self.binance_price_ts: float = 0.0
        self.chainlink_price: float = 0.0
        self.chainlink_price_ts: float = 0.0

        # Window tracking
        self.window_open_price: float = 0.0   # BTC price at window start
        self.window_open_ts: float = 0.0       # Unix timestamp of window start
        self.current_window_ts: int = 0        # Current window identifier (unix ts)
        self.last_traded_window: int = 0       # Prevent double-trading same window

        # Market token cache: {window_ts: {"up_token": str, "down_token": str, "condition_id": str}}
        self.market_cache: dict = {}

        # Performance
        self.windows_traded: int = 0
        self.windows_won: int = 0
        self.windows_skipped: int = 0
        self.total_pnl: float = 0.0

        # ── NEW: Trade buffer for indicators ──
        # Each entry: {"ts": float, "price": float, "qty": float, "is_buy": bool, "usd_volume": float}
        self.trade_buffer: deque = deque(maxlen=10000)  # ~10K trades covers 5+ min

        # Large trade tracker (>$500K)
        self.large_trades: deque = deque(maxlen=100)

        # CVD tracking (updated on each check_for_signal call)
        self.cvd_30s: float = 0.0
        self.cvd_60s: float = 0.0
        self.cvd_120s: float = 0.0

        # Last signal breakdown for logging/dashboard
        self.last_signal_breakdown: dict = {}

        # Signal tracking counters (for monitoring)
        self.signals_generated: int = 0    # Score crossed MIN_SIGNAL_SCORE
        self.signals_blocked: int = 0      # Score crossed but orderbook too expensive
        self.signals_executed: int = 0     # Actually traded

        # WebSocket state
        self._ws_binance = None
        self._ws_rtds = None
        self._running = False

    # ── WINDOW MATH ───────────────────────────────────────────

    @staticmethod
    def current_window_start() -> int:
        """Get the unix timestamp of the current 5-minute window start."""
        now = int(time.time())
        return now - (now % WINDOW_SECONDS)

    @staticmethod
    def next_window_start() -> int:
        """Get the unix timestamp of the next window start."""
        now = int(time.time())
        return now - (now % WINDOW_SECONDS) + WINDOW_SECONDS

    @staticmethod
    def seconds_until_window_close() -> float:
        """Seconds remaining until current window closes."""
        now = time.time()
        window_end = (int(now) - (int(now) % WINDOW_SECONDS)) + WINDOW_SECONDS
        return window_end - now

    @staticmethod
    def window_slug(window_ts: int) -> str:
        """Polymarket market slug for a given window timestamp."""
        return f"{SLUG_PREFIX}{window_ts}"

    # ── PRICE FEEDS ───────────────────────────────────────────

    def best_price(self) -> float:
        """Get the most recent BTC price from any source."""
        # Prefer Binance (faster updates), fall back to Chainlink
        if self.binance_price > 0 and (time.time() - self.binance_price_ts) < 5:
            return self.binance_price
        if self.chainlink_price > 0 and (time.time() - self.chainlink_price_ts) < 30:
            return self.chainlink_price
        return 0.0

    async def _stream_binance(self):
        """Stream BTC/USDT aggregated trades from Binance WebSocket.

        Extracts price, quantity, and trade direction (buy/sell) for:
        - Real-time BTC price tracking
        - CVD (Cumulative Volume Delta) calculation
        - Large trade / whale detection
        """
        while self._running:
            try:
                async with websockets.connect(BINANCE_WS_URL, ping_interval=20) as ws:
                    log.info("Binance aggTrade WebSocket connected")
                    while self._running:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        raw = json.loads(msg)

                        # Combined stream wraps in {"stream": "...", "data": {...}}
                        data = raw.get("data", raw)

                        price = float(data["p"])
                        qty = float(data["q"])
                        is_buyer_maker = data["m"]  # True = taker sold, False = taker bought

                        # Update price feed
                        self.binance_price = price
                        self.binance_price_ts = time.time()

                        # Classify trade direction
                        is_buy = not is_buyer_maker  # taker buy = bullish pressure
                        usd_volume = price * qty

                        now = time.time()
                        trade_record = {
                            "ts": now,
                            "price": price,
                            "qty": qty,
                            "is_buy": is_buy,
                            "usd_volume": usd_volume,
                        }
                        self.trade_buffer.append(trade_record)

                        # Detect large trades (>$500K)
                        if usd_volume >= LARGE_TRADE_THRESHOLD:
                            self.large_trades.append(trade_record)
                            direction = "BUY" if is_buy else "SELL"
                            log.info(
                                f"WHALE {direction}: ${usd_volume:,.0f} | "
                                f"{qty:.4f} BTC @ ${price:,.2f}"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    log.warning(f"Binance WS error: {e}, reconnecting in 2s...")
                    await asyncio.sleep(2)

    async def _poll_chainlink(self):
        """Poll Chainlink BTC/USD price as a backup source (every 10s)."""
        while self._running:
            try:
                # Use CoinGecko as Chainlink-aligned source (both track aggregated prices)
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            price = float(data.get("bitcoin", {}).get("usd", 0))
                            if price > 0:
                                self.chainlink_price = price
                                self.chainlink_price_ts = time.time()
            except asyncio.CancelledError:
                break
            except Exception:
                pass  # Silent fail, Binance is primary

            await asyncio.sleep(10)

    # ── MARKET DISCOVERY ──────────────────────────────────────

    async def _fetch_market_tokens(self, window_ts: int) -> Optional[dict]:
        """Fetch the Up/Down token IDs for a specific 5-minute window."""
        if window_ts in self.market_cache:
            return self.market_cache[window_ts]

        slug = self.window_slug(window_ts)
        url = f"https://gamma-api.polymarket.com/markets?slug={slug}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()

            if not data or not isinstance(data, list) or len(data) == 0:
                # Try CLOB API instead
                return await self._fetch_market_tokens_clob(window_ts)

            market = data[0]
            condition_id = market.get("conditionId", market.get("condition_id", ""))
            tokens_raw = market.get("clobTokenIds", [])
            outcomes_raw = market.get("outcomes", [])

            # These may be JSON strings, parse if needed
            if isinstance(tokens_raw, str):
                tokens = json.loads(tokens_raw)
            else:
                tokens = tokens_raw
            if isinstance(outcomes_raw, str):
                outcomes = json.loads(outcomes_raw)
            else:
                outcomes = outcomes_raw

            if len(tokens) >= 2 and len(outcomes) >= 2:
                result = {"condition_id": condition_id}
                for i, outcome in enumerate(outcomes):
                    if outcome.lower() in ("up", "yes"):
                        result["up_token"] = tokens[i]
                    elif outcome.lower() in ("down", "no"):
                        result["down_token"] = tokens[i]

                # Extract real market prices from Gamma API
                outcome_prices = market.get("outcomePrices", [])
                if isinstance(outcome_prices, str):
                    outcome_prices = json.loads(outcome_prices)
                if len(outcome_prices) >= 2:
                    for i, outcome in enumerate(outcomes):
                        if outcome.lower() in ("up", "yes"):
                            result["up_price"] = float(outcome_prices[i])
                        elif outcome.lower() in ("down", "no"):
                            result["down_price"] = float(outcome_prices[i])

                if "up_token" in result and "down_token" in result:
                    self.market_cache[window_ts] = result
                    return result

        except Exception as e:
            log.debug(f"Gamma API lookup failed for {slug}: {e}")

        return await self._fetch_market_tokens_clob(window_ts)

    async def _fetch_market_tokens_clob(self, window_ts: int) -> Optional[dict]:
        """Fallback: fetch market tokens from CLOB API."""
        slug = self.window_slug(window_ts)
        url = f"https://clob.polymarket.com/markets?slug={slug}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()

            if not data:
                return None

            # CLOB returns single market or list
            market = data[0] if isinstance(data, list) else data
            condition_id = market.get("condition_id", "")
            tokens = market.get("tokens", [])

            result = {"condition_id": condition_id}
            for t in tokens:
                outcome = t.get("outcome", "").lower()
                token_id = t.get("token_id", "")
                if outcome in ("up", "yes"):
                    result["up_token"] = token_id
                elif outcome in ("down", "no"):
                    result["down_token"] = token_id

            if "up_token" in result and "down_token" in result:
                self.market_cache[window_ts] = result
                return result

        except Exception as e:
            log.debug(f"CLOB API lookup failed for {slug}: {e}")

        return None

    async def _get_gamma_price(self, window_ts: int, side: str) -> float:
        """Fetch fresh outcome price from Gamma API for the given side."""
        slug = self.window_slug(window_ts)
        url = f"https://gamma-api.polymarket.com/markets?slug={slug}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status != 200:
                        return 0.0
                    data = await resp.json()

            if not data or not isinstance(data, list) or len(data) == 0:
                return 0.0

            market = data[0]
            outcomes_raw = market.get("outcomes", [])
            prices_raw = market.get("outcomePrices", [])

            if isinstance(outcomes_raw, str):
                outcomes_raw = json.loads(outcomes_raw)
            if isinstance(prices_raw, str):
                prices_raw = json.loads(prices_raw)

            for i, outcome in enumerate(outcomes_raw):
                if outcome.lower() == side.lower() and i < len(prices_raw):
                    price = float(prices_raw[i])
                    return price

        except Exception as e:
            log.debug(f"Gamma price lookup failed for {slug} {side}: {e}")

        return 0.0

    # ── INDICATOR CALCULATIONS ─────────────────────────────────

    def _calculate_cvd(self, window_seconds: int) -> float:
        """
        Calculate Cumulative Volume Delta over a rolling time window.
        CVD = sum(buy_volume_usd) - sum(sell_volume_usd)
        Positive = net buying pressure, Negative = net selling pressure.
        """
        cutoff = time.time() - window_seconds
        cvd = 0.0
        for trade in reversed(self.trade_buffer):
            if trade["ts"] < cutoff:
                break
            if trade["is_buy"]:
                cvd += trade["usd_volume"]
            else:
                cvd -= trade["usd_volume"]
        return cvd

    def _score_cvd(self) -> tuple:
        """
        Calculate CVD-based directional score.
        Returns (score, direction) where score is 0.0-1.0 and direction is "Up" or "Down".

        Uses weighted combination of 30s, 60s, and 120s CVD windows.
        Shorter windows weighted more heavily (more recent = more relevant).
        """
        cvd_30 = self._calculate_cvd(30)
        cvd_60 = self._calculate_cvd(60)
        cvd_120 = self._calculate_cvd(120)

        # Store for logging/dashboard
        self.cvd_30s = cvd_30
        self.cvd_60s = cvd_60
        self.cvd_120s = cvd_120

        # Weighted combination: 50% 30s + 30% 60s + 20% 120s
        weighted_cvd = cvd_30 * 0.50 + cvd_60 * 0.30 + cvd_120 * 0.20

        # Direction
        direction = "Up" if weighted_cvd >= 0 else "Down"
        abs_cvd = abs(weighted_cvd)

        # Map to 0.0-1.0 score
        if abs_cvd < CVD_WEAK_THRESHOLD:
            score = abs_cvd / CVD_WEAK_THRESHOLD * 0.2  # 0.0-0.2
        elif abs_cvd < CVD_MEDIUM_THRESHOLD:
            score = 0.2 + (abs_cvd - CVD_WEAK_THRESHOLD) / (CVD_MEDIUM_THRESHOLD - CVD_WEAK_THRESHOLD) * 0.3  # 0.2-0.5
        elif abs_cvd < CVD_STRONG_THRESHOLD:
            score = 0.5 + (abs_cvd - CVD_MEDIUM_THRESHOLD) / (CVD_STRONG_THRESHOLD - CVD_MEDIUM_THRESHOLD) * 0.3  # 0.5-0.8
        else:
            score = 0.8 + min((abs_cvd - CVD_STRONG_THRESHOLD) / CVD_STRONG_THRESHOLD, 0.2)  # 0.8-1.0

        score = min(score, 1.0)
        return (score, direction)

    def _score_large_trades(self) -> tuple:
        """
        Score based on large trade detection in last 60 seconds.
        Multiple large trades in same direction = strong signal.
        Returns (score, direction).
        """
        cutoff = time.time() - LARGE_TRADE_WINDOW_SECS

        buy_count = 0
        sell_count = 0
        buy_volume = 0.0
        sell_volume = 0.0

        for trade in reversed(self.large_trades):
            if trade["ts"] < cutoff:
                break
            if trade["is_buy"]:
                buy_count += 1
                buy_volume += trade["usd_volume"]
            else:
                sell_count += 1
                sell_volume += trade["usd_volume"]

        # Direction is determined by the dominant side
        if buy_volume > sell_volume:
            direction = "Up"
            dominant_count = buy_count
            dominant_volume = buy_volume
        elif sell_volume > buy_volume:
            direction = "Down"
            dominant_count = sell_count
            dominant_volume = sell_volume
        else:
            return (0.0, "Up")  # No large trades or balanced

        # Scoring: 1 trade = 0.3, 2 same-direction = 0.6, 3+ = 0.8
        score = 0.0
        if dominant_count >= 3:
            score = 0.8
        elif dominant_count == 2:
            score = 0.6
        elif dominant_count == 1:
            score = 0.3

        # Boost for whale-sized trades
        if dominant_volume >= WHALE_TRADE_THRESHOLD * 2:
            score = min(score + 0.2, 1.0)
        elif dominant_volume >= WHALE_TRADE_THRESHOLD:
            score = min(score + 0.1, 1.0)

        return (score, direction)

    def _score_price_delta(self) -> tuple:
        """
        Score based on BTC price delta from window open.
        Returns (score, direction).
        """
        current_price = self.best_price()
        if current_price <= 0 or self.window_open_price <= 0:
            return (0.0, "Up")

        delta = current_price - self.window_open_price
        delta_pct = abs(delta / self.window_open_price) * 100
        direction = "Up" if delta >= 0 else "Down"

        # Score: 0.0 at 0%, 0.5 at 0.05%, 1.0 at 0.15%+
        if delta_pct < 0.01:
            score = 0.0
        elif delta_pct < 0.05:
            score = delta_pct / 0.05 * 0.5  # 0.0-0.5
        elif delta_pct < 0.15:
            score = 0.5 + (delta_pct - 0.05) / 0.10 * 0.5  # 0.5-1.0
        else:
            score = 1.0

        return (score, direction)

    def _calculate_composite_signal(self) -> Optional[dict]:
        """
        Calculate composite signal score from all indicators.
        Returns signal details dict if score exceeds threshold, else None.
        """
        cvd_score, cvd_dir = self._score_cvd()
        delta_score, delta_dir = self._score_price_delta()
        large_score, large_dir = self._score_large_trades()

        # Determine consensus direction via weighted vote
        up_weight = 0.0
        down_weight = 0.0

        for score, direction, weight in [
            (cvd_score, cvd_dir, WEIGHT_CVD),
            (delta_score, delta_dir, WEIGHT_PRICE_DELTA),
            (large_score, large_dir, WEIGHT_LARGE_TRADES),
        ]:
            if direction == "Up":
                up_weight += score * weight
            else:
                down_weight += score * weight

        # Direction = whichever has more weighted score
        if up_weight >= down_weight:
            consensus_direction = "Up"
            directional_score = up_weight
            opposing_score = down_weight
        else:
            consensus_direction = "Down"
            directional_score = down_weight
            opposing_score = up_weight

        # Net score = directional minus opposing (conflicting indicators weaken signal)
        net_score = directional_score - opposing_score * 0.5

        # Agreement bonus: if 2+ active indicators point the same way
        active_indicators = []
        for score, direction in [(cvd_score, cvd_dir), (delta_score, delta_dir), (large_score, large_dir)]:
            if score > 0.1:  # Only count indicators with meaningful signal
                active_indicators.append(direction)

        all_agree = len(active_indicators) >= 2 and len(set(active_indicators)) == 1
        if all_agree:
            net_score += AGREEMENT_BONUS

        net_score = min(net_score, 1.0)

        # Build breakdown for logging and dashboard
        breakdown = {
            "cvd_score": round(cvd_score, 3),
            "cvd_direction": cvd_dir,
            "cvd_30s": round(self.cvd_30s, 0),
            "cvd_60s": round(self.cvd_60s, 0),
            "cvd_120s": round(self.cvd_120s, 0),
            "delta_score": round(delta_score, 3),
            "delta_direction": delta_dir,
            "large_trade_score": round(large_score, 3),
            "large_trade_direction": large_dir,
            "consensus_direction": consensus_direction,
            "all_agree": all_agree,
            "net_score": round(net_score, 3),
            "threshold": MIN_SIGNAL_SCORE,
        }
        self.last_signal_breakdown = breakdown

        if net_score < MIN_SIGNAL_SCORE:
            return None

        return {
            "direction": consensus_direction,
            "score": net_score,
            "breakdown": breakdown,
        }

    @staticmethod
    def _get_max_price_for_score(score: float) -> float:
        """Get the maximum acceptable buy price based on signal score.

        Stronger signals can accept higher prices because the predicted
        win rate justifies the lower ROI.
        """
        for min_score, max_price in PRICE_TIERS:
            if score >= min_score:
                return max_price
        return 0.50  # Fallback: only accept $0.50

    # ── ORDERBOOK CHECK ───────────────────────────────────────

    async def _check_orderbook(self, token_id: str, side: str, max_price: float) -> tuple:
        """
        Check CLOB orderbook for executable asks within max_price.
        Returns (buy_price, available_shares) or (0.0, 0.0) if no viable entry.
        """
        buy_price = 0.0
        available_shares = 0
        try:
            async with aiohttp.ClientSession() as session:
                book_url = f"https://clob.polymarket.com/book?token_id={token_id}"
                async with session.get(book_url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        book = await resp.json()
                        asks = book.get("asks", [])
                        bids = book.get("bids", [])

                        asks_sorted = sorted(asks, key=lambda x: float(x.get("price", 999))) if asks else []
                        bids_sorted = sorted(bids, key=lambda x: float(x.get("price", 0)), reverse=True) if bids else []

                        best_ask = float(asks_sorted[0].get("price", 0)) if asks_sorted else 0
                        best_ask_sz = float(asks_sorted[0].get("size", 0)) if asks_sorted else 0
                        best_bid = float(bids_sorted[0].get("price", 0)) if bids_sorted else 0

                        # Calculate total depth available up to max_price
                        sweepable_depth = 0
                        max_sweep_price = 0
                        for a in asks_sorted:
                            p = float(a.get("price", 0))
                            sz = float(a.get("size", 0))
                            if p > max_price:
                                break
                            sweepable_depth += sz
                            max_sweep_price = p

                        log.info(
                            f"Window {self.current_window_ts} | {side} book: "
                            f"bid=${best_bid:.2f} / ask=${best_ask:.2f} x {best_ask_sz:.0f} | "
                            f"sweepable <=${max_price:.2f}: {sweepable_depth:.0f} shares "
                            f"up to ${max_sweep_price:.2f}"
                        )

                        if best_ask > 0 and best_ask <= max_price and sweepable_depth >= 5:
                            buy_price = max_sweep_price
                            available_shares = sweepable_depth
                        elif best_ask > max_price:
                            log.info(
                                f"... {side} best_ask ${best_ask:.2f} > "
                                f"max ${max_price:.2f} | market already priced in"
                            )
                        else:
                            log.info(f"... {side} insufficient liquidity ({sweepable_depth:.0f} shares)")
                    else:
                        log.warning(f"CLOB /book returned status {resp.status}")
        except Exception as e:
            log.warning(f"Orderbook fetch failed: {e}")

        return (buy_price, available_shares)

    # ── WINDOW TRACKING ───────────────────────────────────────

    def _update_window(self):
        """Track the current window and record its opening price."""
        window_ts = self.current_window_start()

        if window_ts != self.current_window_ts:
            # New window started
            price = self.best_price()
            if price > 0:
                self.window_open_price = price
                self.window_open_ts = time.time()
                self.current_window_ts = window_ts

                # Reset per-window tracking
                self.last_signal_breakdown = {}

                log.info(
                    f"New window {window_ts} | Open price: ${price:,.2f} | "
                    f"Closes in {WINDOW_SECONDS}s | "
                    f"Trade buffer: {len(self.trade_buffer)} trades"
                )
            else:
                log.warning(f"New window {window_ts} but no BTC price available yet")

    # ── SIGNAL GENERATION ─────────────────────────────────────

    async def check_for_signal(self) -> Optional[dict]:
        """
        Check if we should trade the current window using composite scoring.
        Called continuously (every 1s); returns a signal dict when conditions are met.

        Re-evaluates from T-240s to T-60s, giving indicators time to develop.
        """
        self._update_window()

        # Must have a valid window with an open price
        if self.current_window_ts == 0 or self.window_open_price == 0:
            return None

        # Don't double-trade the same window
        if self.current_window_ts == self.last_traded_window:
            return None

        # Check timing: only evaluate in the T-240s to T-60s window
        secs_left = self.seconds_until_window_close()
        if secs_left > ENTRY_SECONDS_BEFORE_CLOSE or secs_left < SKIP_CUTOFF_SECONDS:
            return None

        # Get current BTC price
        current_price = self.best_price()
        if current_price <= 0:
            log.warning("No BTC price available at entry time")
            return None

        # ── COMPOSITE SIGNAL SCORING ──
        composite = self._calculate_composite_signal()

        if composite is None:
            bd = self.last_signal_breakdown
            delta_pct = (current_price - self.window_open_price) / self.window_open_price * 100

            # Only log every ~10 seconds to reduce noise (when secs_left ends in 0)
            if int(secs_left) % 10 == 0:
                log.info(
                    f"Window {self.current_window_ts} | Score {bd.get('net_score', 0):.3f} "
                    f"< {MIN_SIGNAL_SCORE} | CVD={bd.get('cvd_score', 0):.2f}({bd.get('cvd_direction', '?')[0]}) "
                    f"Delta={bd.get('delta_score', 0):.2f}({bd.get('delta_direction', '?')[0]}) "
                    f"Whale={bd.get('large_trade_score', 0):.2f}({bd.get('large_trade_direction', '?')[0]}) | "
                    f"BTC {delta_pct:+.4f}% | T-{secs_left:.0f}s | WAIT"
                )

            # Only mark as skipped at T-60s (give indicators maximum time)
            if secs_left <= SKIP_CUTOFF_SECONDS + 1:
                self.last_traded_window = self.current_window_ts
                self.windows_skipped += 1
                log.info(
                    f"Window {self.current_window_ts} | No signal by T-{SKIP_CUTOFF_SECONDS}s | SKIP"
                )
            return None

        side = composite["direction"]
        score = composite["score"]
        bd = composite["breakdown"]

        self.signals_generated += 1  # Score crossed threshold

        # Dynamic price cap based on signal strength
        max_price = self._get_max_price_for_score(score)

        # Fetch market tokens
        market_info = await self._fetch_market_tokens(self.current_window_ts)
        if not market_info:
            log.warning(f"Could not find market for window {self.current_window_ts}")
            self.last_traded_window = self.current_window_ts
            return None

        token_id = market_info["up_token"] if side == "Up" else market_info["down_token"]
        condition_id = market_info["condition_id"]

        # Check orderbook with dynamic price cap
        buy_price, available_shares = await self._check_orderbook(token_id, side, max_price)

        if buy_price <= 0:
            self.signals_blocked += 1  # Score crossed but orderbook too expensive
            log.info(
                f"Window {self.current_window_ts} | {side} score={score:.3f} "
                f"max_price=${max_price:.2f} | no viable entry | WAIT"
            )
            # Don't skip — try again next second (price/liquidity may change)
            return None

        self.signals_executed += 1  # Actually trading

        # Confidence derived from composite score
        confidence = 0.55 + score * 0.40  # Maps 0.45-1.0 score to 0.73-0.95
        confidence = min(confidence, 0.99)

        # Mark window as traded
        self.last_traded_window = self.current_window_ts

        delta_pct = (current_price - self.window_open_price) / self.window_open_price * 100

        signal = {
            "source": "btc_sniper",
            "type": "multi_indicator_snipe",
            "condition_id": condition_id,
            "token_id": token_id,
            "market_question": f"BTC Up or Down - 5min window {self.current_window_ts}",
            "recommended_outcome": side,
            "outcome": side,
            "market_price": buy_price,
            "confidence": round(confidence, 3),
            "reason": (
                f"BTC {side} | score={score:.3f} | "
                f"CVD={bd['cvd_score']:.2f}({bd['cvd_direction'][0]}) "
                f"Delta={bd['delta_score']:.2f}({bd['delta_direction'][0]}) "
                f"Whale={bd['large_trade_score']:.2f}({bd['large_trade_direction'][0]}) | "
                f"agree={bd['all_agree']} | "
                f"${self.window_open_price:,.2f} -> ${current_price:,.2f} ({delta_pct:+.4f}%) | "
                f"T-{secs_left:.1f}s | ask=${buy_price:.2f} x {available_shares:.0f}"
            ),
            # Extra metadata for the executor
            "_window_ts": self.current_window_ts,
            "_delta_pct": delta_pct,
            "_btc_open": self.window_open_price,
            "_btc_current": current_price,
            "_secs_left": secs_left,
            "_available_shares": available_shares,
            "_signal_score": score,
            "_signal_breakdown": bd,
        }

        log.info(
            f"SIGNAL: {side} | score={score:.3f} | "
            f"CVD={bd['cvd_score']:.2f} Delta={bd['delta_score']:.2f} "
            f"Whale={bd['large_trade_score']:.2f} | "
            f"agree={bd['all_agree']} | ask=${buy_price:.2f} x {available_shares:.0f} | "
            f"max_price=${max_price:.2f} | conf={confidence:.2f} | T-{secs_left:.1f}s"
        )

        return signal

    # ── LIFECYCLE ──────────────────────────────────────────────

    async def start_streams(self):
        """Start the WebSocket price streams."""
        self._running = True
        log.info("Starting BTC price streams...")
        asyncio.create_task(self._stream_binance())
        asyncio.create_task(self._poll_chainlink())

        # Wait for at least one price feed to connect
        for _ in range(50):  # 5 second timeout
            if self.binance_price > 0 or self.chainlink_price > 0:
                price = self.best_price()
                log.info(f"Price feeds active | BTC: ${price:,.2f}")
                return
            await asyncio.sleep(0.1)

        log.warning("Price feeds not ready after 5s, continuing anyway...")

    def stop_streams(self):
        """Stop the WebSocket price streams."""
        self._running = False

    def get_stats(self) -> dict:
        """Return sniper performance stats including indicator data."""
        return {
            "windows_traded": self.windows_traded,
            "windows_won": self.windows_won,
            "windows_skipped": self.windows_skipped,
            "win_rate": (
                round(self.windows_won / self.windows_traded * 100, 1)
                if self.windows_traded > 0 else 0
            ),
            "total_pnl": round(self.total_pnl, 2),
            "current_btc": self.best_price(),
            "current_window": self.current_window_ts,
            "window_open_price": self.window_open_price,
            # Indicator data for dashboard
            "cvd_30s": round(self.cvd_30s, 0),
            "cvd_60s": round(self.cvd_60s, 0),
            "cvd_120s": round(self.cvd_120s, 0),
            "trade_buffer_size": len(self.trade_buffer),
            "large_trades_recent": len([
                t for t in self.large_trades
                if time.time() - t["ts"] < 300
            ]),
            "last_signal_breakdown": self.last_signal_breakdown,
            "signals_generated": self.signals_generated,
            "signals_blocked": self.signals_blocked,
            "signals_executed": self.signals_executed,
        }

    # ── CLEANUP ───────────────────────────────────────────────

    def cleanup_cache(self):
        """Remove old market cache entries and stale trade data."""
        cutoff = self.current_window_start() - WINDOW_SECONDS * 10
        self.market_cache = {
            k: v for k, v in self.market_cache.items() if k > cutoff
        }
        # Prune large_trades older than 5 minutes
        now = time.time()
        while self.large_trades and now - self.large_trades[0]["ts"] > 300:
            self.large_trades.popleft()
