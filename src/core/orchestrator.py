"""
BTC 5-Minute Sniper Orchestrator
==================================
Replaces the old multi-strategy orchestrator.
Runs a tight loop polling for snipe opportunities every second.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import Config
from src.core.polymarket_client import PolymarketClient
from src.core.live_executor import LiveExecutor
from src.modules.btc_sniper import BTCSniper
from src.utils.logger import setup_logger

log = setup_logger("orchestrator")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
STATS_FILE = DATA_DIR / "sniper_stats.json"


class TradingOrchestrator:
    """
    Single-purpose orchestrator for BTC 5-minute sniper.
    Runs continuously, checking for signals every second.
    """

    def __init__(self):
        self.client = PolymarketClient()

        # Executor
        if Config.SIMULATION_MODE:
            from src.core.simulator import PaperTrader
            self.executor = PaperTrader(self.client)
        else:
            self.executor = LiveExecutor(self.client)

        self.sniper = BTCSniper()
        self.is_running = False

        # Stats
        self.trades_executed = 0
        self.trades_failed = 0
        self.cumulative_pnl = 0.0
        self._load_stats()

    # ── BET SIZING ────────────────────────────────────────────

    def calculate_bet_size(self) -> float:
        """
        Dynamic bet sizing:
        - Base: $50 (configurable)
        - Scale up: add 10% of cumulative profit
        - Cap: never more than 5% of balance
        """
        base = Config.BASE_BET_SIZE
        profit_bonus = max(0, self.cumulative_pnl * Config.PROFIT_SCALE_PCT / 100)
        bet = base + profit_bonus

        # Cap at % of balance
        max_bet = self.executor.balance * Config.MAX_BET_PCT_OF_BALANCE / 100
        bet = min(bet, max_bet)

        # Floor at $5
        bet = max(bet, 5.0)

        return round(bet, 2)

    # ── STATS PERSISTENCE ─────────────────────────────────────

    def _load_stats(self):
        """Load cumulative stats from disk."""
        DATA_DIR.mkdir(exist_ok=True)
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE) as f:
                    stats = json.load(f)
                    self.cumulative_pnl = stats.get("cumulative_pnl", 0.0)
                    self.trades_executed = stats.get("trades_executed", 0)
                    self.trades_failed = stats.get("trades_failed", 0)
                    self.sniper.windows_traded = stats.get("windows_traded", 0)
                    self.sniper.windows_won = stats.get("windows_won", 0)
                    self.sniper.windows_skipped = stats.get("windows_skipped", 0)
            except Exception as e:
                log.warning(f"Could not load stats: {e}")

    def _save_stats(self):
        """Save cumulative stats to disk."""
        DATA_DIR.mkdir(exist_ok=True)
        try:
            with open(STATS_FILE, "w") as f:
                sniper_stats = self.sniper.get_stats()
                json.dump({
                    "cumulative_pnl": round(self.cumulative_pnl, 2),
                    "trades_executed": self.trades_executed,
                    "trades_failed": self.trades_failed,
                    "windows_traded": self.sniper.windows_traded,
                    "windows_won": self.sniper.windows_won,
                    "windows_skipped": self.sniper.windows_skipped,
                    "balance": round(self.executor.balance, 2),
                    "btc_price": round(self.sniper.best_price(), 2),
                    "window_open_price": round(self.sniper.window_open_price, 2),
                    "current_window": self.sniper.current_window_ts,
                    "base_bet": Config.BASE_BET_SIZE,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    # Indicator data for dashboard
                    "cvd_30s": sniper_stats.get("cvd_30s", 0),
                    "cvd_60s": sniper_stats.get("cvd_60s", 0),
                    "cvd_120s": sniper_stats.get("cvd_120s", 0),
                    "trade_buffer_size": sniper_stats.get("trade_buffer_size", 0),
                    "large_trades_recent": sniper_stats.get("large_trades_recent", 0),
                    "last_signal_breakdown": sniper_stats.get("last_signal_breakdown", {}),
                    # Signal tracking counters
                    "signals_generated": sniper_stats.get("signals_generated", 0),
                    "signals_blocked": sniper_stats.get("signals_blocked", 0),
                    "signals_executed": sniper_stats.get("signals_executed", 0),
                }, f, indent=2, default=str)
        except Exception as e:
            log.warning(f"Could not save stats: {e}")

    # ── TRADE EXECUTION ───────────────────────────────────────

    async def _execute_signal(self, signal: dict):
        """Execute a sniper signal with dynamic bet sizing."""
        bet_size = self.calculate_bet_size()

        if bet_size > self.executor.balance:
            log.warning(
                f"Insufficient balance: ${self.executor.balance:.2f} < bet ${bet_size:.2f}"
            )
            return

        # Build decision (compatible with LiveExecutor.execute_trade)
        decision = {
            "approved": True,
            "signal": signal,
            "position_size_usdc": bet_size,
            "max_loss_usdc": bet_size,
        }

        log.info(
            f"EXECUTING: {signal['outcome']} | "
            f"Bet: ${bet_size:.2f} | Balance: ${self.executor.balance:.2f}"
        )

        result = await self.executor.execute_trade(decision)

        if result.get("success"):
            self.trades_executed += 1
            self.sniper.windows_traded += 1
            log.info(f"Trade #{self.trades_executed} placed successfully")
        else:
            self.trades_failed += 1
            reason = result.get("reason", "unknown")
            log.warning(f"Trade failed: {reason}")

        self._save_stats()

    # ── POSITION RESOLUTION ───────────────────────────────────

    async def _check_resolved_positions(self):
        """
        Check if any open positions have resolved.
        For 5-minute markets, resolution happens within ~2 minutes after window close.
        """
        if not self.executor.positions:
            return

        exits = await self.executor.check_exits()
        if exits:
            for ex in exits:
                pnl = ex.get("pnl", 0)
                self.cumulative_pnl += pnl
                if pnl > 0:
                    self.sniper.windows_won += 1

                outcome = "WON" if pnl > 0 else "LOST"
                log.info(
                    f"Position {outcome}: ${pnl:+.2f} | "
                    f"Cumulative P&L: ${self.cumulative_pnl:+.2f}"
                )

            self._save_stats()

    # ── MAIN LOOP ─────────────────────────────────────────────

    async def initialize(self):
        """Startup banner and validation."""
        log.info("=" * 55)
        log.info("BTC 5-MINUTE SNIPER BOT")
        log.info(f"Mode: {'SIMULATION' if Config.SIMULATION_MODE else 'LIVE TRADING'}")
        log.info(f"Wallet: {Config.WALLET_ADDRESS}")
        log.info(f"Base bet: ${Config.BASE_BET_SIZE} | Scale: +{Config.PROFIT_SCALE_PCT}% of profit")
        log.info(f"Entry: T-{Config.ENTRY_SECONDS_BEFORE_CLOSE}s | Min delta: {Config.MIN_PRICE_DELTA_PCT}%")
        log.info(f"Balance: ${self.executor.balance:.2f}")
        if self.trades_executed > 0:
            log.info(
                f"Resuming: {self.trades_executed} trades | "
                f"P&L: ${self.cumulative_pnl:+.2f} | "
                f"Win rate: {self.sniper.windows_won}/{self.sniper.windows_traded}"
            )
        log.info("=" * 55)

        errors = Config.validate()
        if errors:
            for e in errors:
                log.error(f"Config error: {e}")
            if not Config.SIMULATION_MODE:
                raise ValueError("Cannot start live trading with config errors")

    async def run(self, interval_minutes: int = 0, max_cycles: int = 0):
        """
        Main loop. Runs continuously, checking every second.
        interval_minutes is ignored (kept for CLI compatibility).
        """
        self.is_running = True
        await self.initialize()

        # Start price streams
        await self.sniper.start_streams()

        # Wait a moment for prices to stabilize
        await asyncio.sleep(2)

        cycle = 0
        last_summary_ts = 0
        last_stats_save_ts = 0

        try:
            while self.is_running:
                cycle += 1

                # Check for resolved positions every cycle
                await self._check_resolved_positions()

                # Check for new snipe signal
                signal = await self.sniper.check_for_signal()
                if signal:
                    await self._execute_signal(signal)

                # Save stats every 30s (for dashboard live data)
                now_ts = int(datetime.now(timezone.utc).timestamp())
                if now_ts - last_stats_save_ts >= 30:
                    self._save_stats()
                    last_stats_save_ts = now_ts

                # Print summary every 5 minutes
                if now_ts - last_summary_ts >= 300:
                    self._print_summary()
                    last_summary_ts = now_ts
                    self.sniper.cleanup_cache()

                # Max cycles (for testing)
                if max_cycles > 0 and cycle >= max_cycles:
                    log.info(f"Completed {max_cycles} cycles. Stopping.")
                    break

                # Poll every second
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            log.info("Keyboard interrupt - shutting down...")

        await self.shutdown()

    async def run_once(self):
        """Run until one trade is placed or one window passes."""
        await self.initialize()
        await self.sniper.start_streams()
        await asyncio.sleep(2)

        log.info("Running in single-shot mode (will trade next window)...")

        for _ in range(400):  # ~6.5 minutes max wait
            await self._check_resolved_positions()
            signal = await self.sniper.check_for_signal()
            if signal:
                await self._execute_signal(signal)
                break
            await asyncio.sleep(1)

        self._print_summary()
        await self.shutdown()

    def _print_summary(self):
        """Print current status."""
        mode = "SIM" if Config.SIMULATION_MODE else "LIVE"
        stats = self.sniper.get_stats()
        bet = self.calculate_bet_size()

        log.info(f"\n{'='*50}")
        log.info(f"[{mode}] BTC SNIPER STATUS")
        log.info(f"Balance: ${self.executor.balance:.2f} | Next bet: ${bet:.2f}")
        log.info(
            f"Trades: {self.trades_executed} | "
            f"Won: {self.sniper.windows_won} | "
            f"Skipped: {self.sniper.windows_skipped}"
        )
        if self.trades_executed > 0:
            wr = self.sniper.windows_won / self.trades_executed * 100
            log.info(f"Win rate: {wr:.1f}% | P&L: ${self.cumulative_pnl:+.2f}")
        log.info(
            f"BTC: ${stats['current_btc']:,.2f} | "
            f"Window open: ${stats['window_open_price']:,.2f}"
        )
        open_positions = len([
            p for p in self.executor.positions if p.get("status") == "open"
        ])
        if open_positions:
            log.info(f"Open positions: {open_positions} (awaiting resolution)")
        log.info(f"{'_'*50}")

    async def shutdown(self):
        """Clean shutdown."""
        log.info("Shutting down...")
        self.sniper.stop_streams()
        self._save_stats()
        self.is_running = False
        await self.client.close()
        log.info("Shutdown complete.")

    def stop(self):
        self.is_running = False
