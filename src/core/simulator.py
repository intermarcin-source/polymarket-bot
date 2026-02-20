import json
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from src.core.polymarket_client import PolymarketClient
from src.utils.logger import setup_logger

log = setup_logger("simulator")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SIM_TRADES_FILE = DATA_DIR / "sim_trades.json"
SIM_PORTFOLIO_FILE = DATA_DIR / "sim_portfolio.json"

# ── Exit Strategy Thresholds ─────────────────────────────
TAKE_PROFIT_PCT = 20.0      # sell when +20% gain
STOP_LOSS_PCT = -25.0       # sell when -25% loss
TRAILING_STOP_PCT = -10.0   # after hitting +10%, trail by 10%
TRAILING_ACTIVATE_PCT = 10.0  # activate trailing stop after +10%


class PaperTrader:
    """
    Simulated trading engine with active exit strategies.
    Doesn't just hold until resolution - takes profits, cuts losses,
    and recycles capital into the next opportunity.
    """

    def __init__(self, client: PolymarketClient, starting_balance: float = 2000.0):
        self.client = client
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.positions: list[dict] = []
        self.closed_trades: list[dict] = []
        self.trade_history: list[dict] = []
        self.total_pnl = 0.0
        self._load_state()

    def _load_state(self):
        DATA_DIR.mkdir(exist_ok=True)
        if SIM_PORTFOLIO_FILE.exists():
            with open(SIM_PORTFOLIO_FILE) as f:
                state = json.load(f)
                self.balance = state.get("balance", self.starting_balance)
                self.positions = state.get("positions", [])
                self.closed_trades = state.get("closed_trades", [])
                self.total_pnl = state.get("total_pnl", 0.0)
                self.starting_balance = state.get("starting_balance", self.starting_balance)

        if SIM_TRADES_FILE.exists():
            with open(SIM_TRADES_FILE) as f:
                self.trade_history = json.load(f)

    def _save_state(self):
        DATA_DIR.mkdir(exist_ok=True)
        with open(SIM_PORTFOLIO_FILE, "w") as f:
            json.dump({
                "balance": self.balance,
                "positions": self.positions,
                "closed_trades": self.closed_trades,
                "total_pnl": self.total_pnl,
                "starting_balance": self.starting_balance,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2, default=str)

        with open(SIM_TRADES_FILE, "w") as f:
            json.dump(self.trade_history, f, indent=2, default=str)

    # ── TOKEN ID RESOLUTION ──────────────────────────────────

    async def _resolve_token_id(self, condition_id: str, outcome: str) -> str:
        """Resolve a token_id from condition_id + outcome via Gamma API."""
        if not condition_id:
            return ""
        try:
            market = await self.client.get_market(condition_id)
            if market:
                tokens = market.get("tokens", [])
                # Match outcome name (case-insensitive)
                for t in tokens:
                    if t.get("outcome", "").lower() == outcome.lower():
                        tid = t.get("token_id", "")
                        if tid:
                            log.info(f"Resolved token_id for '{outcome}': {tid[:20]}...")
                            return tid
                # If no exact match, try first token as fallback
                if tokens and len(tokens) == 2:
                    # Binary market: Yes/No - pick the matching one
                    for t in tokens:
                        if t.get("outcome", "").lower() == outcome.lower():
                            return t.get("token_id", "")
                    # Last resort: return first token
                    log.warning(f"No exact outcome match for '{outcome}', tokens: {[t.get('outcome') for t in tokens]}")
        except Exception as e:
            log.debug(f"Token ID resolution failed for {condition_id[:20]}: {e}")
        return ""

    # ── BUY ──────────────────────────────────────────────────

    async def execute_trade(self, decision: dict) -> dict:
        """Execute a simulated BUY."""
        signal = decision["signal"]
        size_usdc = decision["position_size_usdc"]

        if size_usdc > self.balance:
            log.warning(f"Insufficient sim balance: ${self.balance:.2f} < ${size_usdc:.2f}")
            return {"success": False, "reason": "Insufficient balance"}

        token_id = signal.get("token_id", "")
        market_price = signal.get("market_price", 0)
        outcome = signal.get("recommended_outcome", signal.get("outcome", ""))
        source = signal.get("source", "")

        # CRITICAL: Resolve token_id if missing - needed for live price tracking
        if not token_id and signal.get("condition_id"):
            token_id = await self._resolve_token_id(signal["condition_id"], outcome)

        # For btc_sniper signals, trust the CLOB orderbook price from the signal
        # (it was fetched at the exact moment of signal generation)
        if source != "btc_sniper" and token_id:
            prices = await self.client.get_prices([token_id])
            if token_id in prices:
                market_price = prices[token_id]

        if not market_price or market_price <= 0:
            market_price = 0.5

        # Skip unrealistic prices where there's no real orderbook liquidity
        # Below 5 cents = likely no sellers, above 97 cents = no upside
        if market_price < 0.05:
            return {"success": False, "reason": f"Price too low ({market_price:.4f}), no liquidity"}

        if market_price > 0.97:
            return {"success": False, "reason": f"Price too high ({market_price:.4f}), no upside"}

        if market_price >= 1:
            return {"success": False, "reason": f"Invalid price: {market_price}"}

        shares = size_usdc / market_price

        # Cap shares to available orderbook depth (btc_sniper provides this)
        available = signal.get("_available_shares")
        if available and shares > available:
            shares = available
            size_usdc = shares * market_price
            log.info(f"Capped shares to orderbook depth: {shares:.0f} @ ${market_price:.4f} = ${size_usdc:.2f}")

        trade = {
            "id": f"sim_{len(self.trade_history) + 1}",
            "condition_id": signal.get("condition_id", ""),
            "market_question": signal.get("market_question", ""),
            "outcome": signal.get("recommended_outcome", signal.get("outcome", "")),
            "token_id": token_id,
            "side": "buy",
            "entry_price": market_price,
            "price": market_price,
            "current_price": market_price,
            "shares": shares,
            "size_usdc": size_usdc,
            "source": signal.get("source", ""),
            "signal_type": signal.get("type", ""),
            "confidence": signal.get("confidence", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "open",
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "peak_pnl_pct": 0.0,   # track highest gain for trailing stop
            "exit_reason": "",
        }

        self.balance -= size_usdc
        self.positions.append(trade)
        self.trade_history.append(trade)
        self._save_state()

        log.info(
            f"[BUY] {shares:.2f} shares of '{trade['outcome']}' "
            f"@ ${market_price:.4f} = ${size_usdc:.2f} "
            f"| {signal.get('market_question', '?')[:50]} "
            f"| Source: {signal.get('source', '?')}"
        )

        return {"success": True, "trade": trade}

    # ── SELL / EXIT ──────────────────────────────────────────

    async def sell_position(self, position: dict, reason: str) -> dict:
        """Sell an open position at current market price."""
        token_id = position.get("token_id", "")
        current_price = position.get("current_price", position["entry_price"])

        # Get fresh price
        if token_id:
            prices = await self.client.get_prices([token_id])
            if token_id in prices:
                current_price = prices[token_id]

        # Calculate proceeds
        proceeds = position["shares"] * current_price
        cost = position["size_usdc"]
        profit = proceeds - cost
        pnl_pct = (current_price / position["entry_price"] - 1) * 100 if position["entry_price"] > 0 else 0

        # Update position
        position["status"] = "sold"
        position["exit_price"] = current_price
        position["exit_reason"] = reason
        position["pnl"] = profit
        position["pnl_pct"] = pnl_pct
        position["closed_at"] = datetime.now(timezone.utc).isoformat()

        # Return capital + profit
        self.balance += proceeds
        self.total_pnl += profit

        # Move to closed trades
        self.closed_trades.append(position)

        pnl_style = "PROFIT" if profit >= 0 else "LOSS"
        log.info(
            f"[SELL - {pnl_style}] '{position['outcome']}' "
            f"@ ${current_price:.4f} (entry: ${position['entry_price']:.4f}) "
            f"| PnL: ${profit:+.2f} ({pnl_pct:+.1f}%) "
            f"| Reason: {reason} "
            f"| {position['market_question'][:45]}"
        )

        self._save_state()
        return {"success": True, "profit": profit, "pnl_pct": pnl_pct}

    # ── FIX MISSING TOKEN IDS ─────────────────────────────────

    async def _backfill_token_ids(self):
        """Resolve token_ids for any existing positions that are missing them."""
        fixed = 0
        for pos in self.positions:
            if pos.get("status") != "open":
                continue
            if pos.get("token_id"):
                continue  # already has one

            condition_id = pos.get("condition_id", "")
            outcome = pos.get("outcome", "")
            if condition_id and outcome:
                token_id = await self._resolve_token_id(condition_id, outcome)
                if token_id:
                    pos["token_id"] = token_id
                    fixed += 1
                await asyncio.sleep(0.3)  # rate limit

        if fixed:
            log.info(f"Backfilled token_ids for {fixed} positions")
            self._save_state()

    # ── EXIT STRATEGY ENGINE ─────────────────────────────────

    async def check_exits(self) -> list[dict]:
        """
        Core exit strategy. Checks every open position and sells if:
        1. Take profit: gained >= 20%
        2. Stop loss: lost >= 30%
        3. Trailing stop: was up 10%+, now dropped 10% from peak
        4. Market resolved: event ended

        Returns list of closed trades this cycle.
        """
        # First: fix any positions missing token_ids
        needs_backfill = any(
            not p.get("token_id") and p.get("status") == "open"
            for p in self.positions
        )
        if needs_backfill:
            log.info("Some positions missing token_ids, resolving...")
            await self._backfill_token_ids()

        exits = []
        remaining = []

        for pos in self.positions:
            if pos["status"] != "open":
                remaining.append(pos)
                continue

            try:
                token_id = pos.get("token_id", "")
                entry_price = pos.get("entry_price", pos.get("price", 0.5))

                # Get current price
                current_price = entry_price
                if token_id:
                    try:
                        prices = await self.client.get_prices([token_id])
                        if token_id in prices:
                            current_price = prices[token_id]
                    except Exception:
                        pass  # keep entry_price if orderbook fails

                exit_reason = None

                # ── STEP 1: Check market resolution FIRST ──
                # This must come before price-based exits because resolved
                # markets have empty orderbooks that return $0.001 (fake price).
                condition_id = pos.get("condition_id", "")
                if condition_id:
                    try:
                        market = await self.client.get_market(condition_id)
                    except Exception:
                        market = None
                    if market and market.get("closed"):
                        winning = ""
                        for t in market.get("tokens", []):
                            if t.get("winner"):
                                winning = t.get("outcome", "")
                                break
                        if not winning:
                            winning = market.get("winningOutcome", "")
                        if winning:
                            if pos["outcome"].lower() == winning.lower():
                                current_price = 1.0
                                exit_reason = "Market resolved - WON"
                            else:
                                current_price = 0.0
                                exit_reason = "Market resolved - LOST"

                # ── STEP 2: Price-based exits (only if not resolved) ──
                # If price is suspiciously low ($0.001), don't trust it —
                # the orderbook is probably empty, not a real price drop.
                if not exit_reason and current_price < 0.02:
                    log.debug(
                        f"Ignoring suspicious price ${current_price:.4f} for "
                        f"'{pos.get('market_question', '?')[:40]}' — likely empty orderbook"
                    )
                    # Keep entry price so P&L stays at 0% instead of -99%
                    current_price = entry_price

                pos["current_price"] = current_price
                pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0
                pos["pnl"] = (current_price - entry_price) * pos["shares"]
                pos["pnl_pct"] = pnl_pct

                # Track peak for trailing stop
                if pnl_pct > pos.get("peak_pnl_pct", 0):
                    pos["peak_pnl_pct"] = pnl_pct

                if not exit_reason:
                    # TAKE PROFIT
                    if pnl_pct >= TAKE_PROFIT_PCT:
                        exit_reason = f"Take profit ({pnl_pct:+.1f}% >= {TAKE_PROFIT_PCT}%)"

                    # STOP LOSS
                    elif pnl_pct <= STOP_LOSS_PCT:
                        exit_reason = f"Stop loss ({pnl_pct:+.1f}% <= {STOP_LOSS_PCT}%)"

                    # TRAILING STOP
                    elif pos.get("peak_pnl_pct", 0) >= TRAILING_ACTIVATE_PCT:
                        drop_from_peak = pnl_pct - pos["peak_pnl_pct"]
                        if drop_from_peak <= TRAILING_STOP_PCT:
                            exit_reason = (
                                f"Trailing stop (peak: {pos['peak_pnl_pct']:+.1f}%, "
                                f"now: {pnl_pct:+.1f}%, drop: {drop_from_peak:.1f}%)"
                            )

                # Execute exit
                if exit_reason:
                    result = await self.sell_position(pos, exit_reason)
                    exits.append({**pos, "exit_result": result})
                else:
                    remaining.append(pos)

            except Exception as e:
                log.warning(f"Error checking position {pos.get('id','?')}: {e}")
                remaining.append(pos)  # keep position, try again next cycle

        self.positions = remaining
        self._save_state()

        if exits:
            total_exit_pnl = sum(e.get("pnl", 0) for e in exits)
            log.info(
                f"[EXIT SUMMARY] Closed {len(exits)} positions | "
                f"Combined PnL: ${total_exit_pnl:+.2f} | "
                f"Capital freed: ${sum(e.get('size_usdc', 0) for e in exits):.2f}"
            )

        return exits

    async def update_positions(self):
        """Update all open positions with current market prices."""
        updated = 0
        for pos in self.positions:
            if pos["status"] != "open":
                continue

            token_id = pos.get("token_id", "")
            if not token_id:
                continue

            prices = await self.client.get_prices([token_id])
            if token_id in prices:
                current_price = prices[token_id]
                entry_price = pos.get("entry_price", pos.get("price", 0.5))
                pos["current_price"] = current_price
                pos["pnl"] = (current_price - entry_price) * pos["shares"]
                pos["pnl_pct"] = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0

                if pos["pnl_pct"] > pos.get("peak_pnl_pct", 0):
                    pos["peak_pnl_pct"] = pos["pnl_pct"]
                updated += 1

        if updated:
            log.info(f"Updated prices for {updated}/{len([p for p in self.positions if p['status'] == 'open'])} open positions")
        self._save_state()

    def get_performance(self) -> dict:
        """Get overall simulation performance metrics."""
        open_trades = [p for p in self.positions if p["status"] == "open"]
        sold_trades = self.closed_trades
        won_trades = [t for t in sold_trades if t.get("pnl", 0) > 0]
        lost_trades = [t for t in sold_trades if t.get("pnl", 0) <= 0]
        total_closed = len(sold_trades)

        unrealized_pnl = sum(p.get("pnl", 0) for p in open_trades)
        realized_pnl = sum(t.get("pnl", 0) for t in sold_trades)

        # By source
        by_source: dict[str, dict] = {}
        for trade in sold_trades:
            src = trade.get("source", "unknown")
            if src not in by_source:
                by_source[src] = {"trades": 0, "pnl": 0, "wins": 0, "losses": 0}
            by_source[src]["trades"] += 1
            pnl = trade.get("pnl", 0)
            by_source[src]["pnl"] += pnl
            if pnl > 0:
                by_source[src]["wins"] += 1
            else:
                by_source[src]["losses"] += 1

        # By exit reason
        by_exit: dict[str, int] = {}
        for trade in sold_trades:
            reason = trade.get("exit_reason", "unknown")
            # Simplify reason
            if "Take profit" in reason:
                key = "Take Profit"
            elif "Stop loss" in reason:
                key = "Stop Loss"
            elif "Trailing stop" in reason:
                key = "Trailing Stop"
            elif "resolved" in reason:
                key = "Market Resolved"
            else:
                key = reason
            by_exit[key] = by_exit.get(key, 0) + 1

        total_value = self.balance + sum(
            p["shares"] * p.get("current_price", p.get("entry_price", 0.5))
            for p in open_trades
        )

        return {
            "starting_balance": self.starting_balance,
            "current_balance": round(self.balance, 2),
            "total_value": round(total_value, 2),
            "total_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "return_pct": round((total_value / self.starting_balance - 1) * 100, 2),
            "total_trades": len(self.trade_history),
            "open_positions": len(open_trades),
            "closed_trades": total_closed,
            "wins": len(won_trades),
            "losses": len(lost_trades),
            "win_rate": round(len(won_trades) / total_closed * 100, 1) if total_closed > 0 else 0,
            "avg_win": round(sum(t["pnl"] for t in won_trades) / len(won_trades), 2) if won_trades else 0,
            "avg_loss": round(sum(t["pnl"] for t in lost_trades) / len(lost_trades), 2) if lost_trades else 0,
            "by_source": by_source,
            "by_exit_reason": by_exit,
        }

    def reset(self):
        """Reset simulation to starting state."""
        self.balance = self.starting_balance
        self.positions = []
        self.closed_trades = []
        self.trade_history = []
        self.total_pnl = 0.0
        self._save_state()
        log.info("Simulation reset to starting state")
