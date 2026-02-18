import asyncio
from datetime import datetime, timezone
from src.core.config import Config
from src.core.polymarket_client import PolymarketClient
from src.core.risk_manager import RiskManager
from src.core.simulator import PaperTrader
from src.modules.whale_tracker import WhaleTracker
from src.modules.arbitrage_scanner import ArbitrageScanner
from src.modules.market_maker import MarketMaker
from src.modules.mean_reversion import MeanReversionTrader
from src.modules.negrisk_scanner import NegRiskScanner
from src.utils.logger import setup_logger

log = setup_logger("orchestrator")


class TradingOrchestrator:
    """
    Main decision engine. Multi-strategy trading bot:
    1. Whale Copy Trading - mirror profitable wallets
    2. Single-Market Arbitrage - YES + NO < $1.00
    3. Market Making - spread capture on wide bid-ask
    4. Mean Reversion - trade overreaction bounces
    5. NegRisk Rebalancing - multi-outcome probability arbitrage
    6. Bot Copy Trading - mirror profitable automated traders (existing)

    Each cycle:
    1. CHECK EXITS first - take profits, cut losses, free capital
    2. SCAN for new signals from ALL enabled strategies
    3. ENTER new positions with freed capital (highest confidence first)
    """

    def __init__(self):
        self.client = PolymarketClient()
        self.risk_manager = RiskManager()

        # Unified executor: PaperTrader for sim, LiveExecutor for real money
        if Config.SIMULATION_MODE:
            self.executor = PaperTrader(self.client)
        else:
            from src.core.live_executor import LiveExecutor
            self.executor = LiveExecutor(self.client)

        # Backward compatibility alias
        self.simulator = self.executor

        # Strategy modules (initialized based on config toggles)
        self.whale_tracker = WhaleTracker(self.client) if Config.ENABLE_WHALE_TRACKER else None
        self.arbitrage_scanner = ArbitrageScanner(self.client) if Config.ENABLE_ARBITRAGE else None
        self.market_maker = MarketMaker(self.client) if Config.ENABLE_MARKET_MAKER else None
        self.mean_reversion = MeanReversionTrader(self.client) if Config.ENABLE_MEAN_REVERSION else None
        self.negrisk_scanner = NegRiskScanner(self.client) if Config.ENABLE_NEGRISK else None

        self.is_running = False
        self.cycle_count = 0

    def _enabled_strategies(self) -> list[str]:
        """Return list of enabled strategy names."""
        strategies = []
        if self.whale_tracker:
            strategies.append("Whale Copy")
        if self.arbitrage_scanner:
            strategies.append("Arbitrage")
        if self.market_maker:
            strategies.append("Market Maker")
        if self.mean_reversion:
            strategies.append("Mean Reversion")
        if self.negrisk_scanner:
            strategies.append("NegRisk")
        return strategies

    async def initialize(self):
        """Run initial setup: discover whales, log config."""
        strategies = self._enabled_strategies()
        log.info("=" * 60)
        log.info("POLYMARKET MULTI-STRATEGY TRADING BOT")
        log.info(f"Mode: {'SIMULATION' if Config.SIMULATION_MODE else 'LIVE TRADING'}")
        log.info(f"Wallet: {Config.WALLET_ADDRESS}")
        log.info(f"Strategies: {', '.join(strategies)} ({len(strategies)} active)")
        log.info(f"Max bet: ${Config.MAX_BET_SIZE} | Max exposure: ${Config.MAX_PORTFOLIO_EXPOSURE}")
        log.info(f"Exit rules: TP +20% | SL -30% | Trail -10% from peak")
        log.info("=" * 60)

        errors = Config.validate()
        if errors:
            for e in errors:
                log.error(f"Config error: {e}")
            if not Config.SIMULATION_MODE:
                raise ValueError("Cannot start live trading with config errors")

        if self.whale_tracker:
            log.info("Discovering whale wallets...")
            await self.whale_tracker.auto_add_whales()

        log.info("Initialization complete!")

    async def _collect_signals(self) -> list[dict]:
        """
        Collect signals from ALL enabled strategies in parallel where possible.
        Returns combined list sorted by confidence.
        """
        all_signals = []
        tasks = []
        task_names = []

        # Queue up all enabled strategy scans
        if self.whale_tracker:
            tasks.append(self.whale_tracker.scan_for_signals())
            task_names.append("whale")

        if self.arbitrage_scanner:
            tasks.append(self.arbitrage_scanner.scan_for_signals())
            task_names.append("arbitrage")

        if self.market_maker:
            tasks.append(self.market_maker.scan_for_signals())
            task_names.append("market_maker")

        if self.mean_reversion:
            tasks.append(self.mean_reversion.scan_for_signals())
            task_names.append("mean_reversion")

        if self.negrisk_scanner:
            tasks.append(self.negrisk_scanner.scan_for_signals())
            task_names.append("negrisk")

        if not tasks:
            log.warning("No strategies enabled!")
            return []

        # Run all strategies in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                log.error(f"{name} strategy error: {result}")
            elif result:
                all_signals.extend(result)
                log.info(f"  {name}: {len(result)} signals")

        return all_signals

    async def run_cycle(self):
        """
        One full trading cycle:
        STEP 1: Check exits on all open positions (take profit / stop loss)
        STEP 2: Sync risk manager with actual positions
        STEP 3: Collect new signals from ALL strategies
        STEP 4: Enter new trades with available capital
        """
        self.cycle_count += 1
        log.info(f"\n{'='*50}")
        log.info(f"CYCLE #{self.cycle_count} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        log.info(f"{'='*50}")

        # -- STEP 1: CHECK EXITS (both sim and live) --
        if self.executor.positions:
            log.info(f"Checking {len(self.executor.positions)} open positions for exits...")
            exits = await self.executor.check_exits()
            if exits:
                for ex in exits:
                    self.risk_manager.close_position(
                        ex.get("condition_id", ""),
                        ex.get("pnl", 0),
                    )
                log.info(f"Exited {len(exits)} positions, capital recycled for new trades")
            else:
                log.info("No exit triggers hit, all positions holding")
                await self.executor.update_positions()

        # -- STEP 2: SYNC RISK MANAGER --
        open_value = sum(
            p.get("size_usdc", 0) for p in self.executor.positions if p.get("status") == "open"
        )
        self.risk_manager.portfolio_value = self.executor.balance + open_value
        self.risk_manager.total_exposure = open_value
        self.risk_manager.open_positions = [
            {"condition_id": p.get("condition_id"), "position_size_usdc": p.get("size_usdc", 0)}
            for p in self.executor.positions if p.get("status") == "open"
        ]

        available = self.risk_manager.available_capital
        log.info(
            f"Portfolio: ${self.risk_manager.portfolio_value:.2f} | "
            f"Exposure: ${open_value:.2f}/${self.risk_manager.max_exposure:.2f} | "
            f"Max bet: ${self.risk_manager.max_bet_size:.2f}"
        )

        if available < 5:
            log.info("Portfolio fully deployed, skipping signal scan (waiting for exits)")
            return

        # -- STEP 3: COLLECT SIGNALS (ALL STRATEGIES) --
        log.info("Scanning all strategies for signals...")
        all_signals = await self._collect_signals()

        signal_counts = {}
        for s in all_signals:
            src = s.get("source", "unknown")
            signal_counts[src] = signal_counts.get(src, 0) + 1

        counts_str = " | ".join(f"{k}: {v}" for k, v in signal_counts.items())
        log.info(f"Total signals: {len(all_signals)} ({counts_str})")

        if not all_signals:
            log.info("No new signals this cycle.")
            return

        # -- STEP 4: ENTER NEW TRADES --
        # Sort by confidence (highest first) - arb signals get priority due to lower risk
        all_signals.sort(key=lambda s: (
            1 if s.get("source") == "arbitrage_scanner" else 0,  # arb gets priority
            s.get("confidence", 0)
        ), reverse=True)

        executed = 0
        rejected_streak = 0
        by_source = {}

        for signal in all_signals:
            if self.risk_manager.available_capital < 1:
                log.info(f"Portfolio fully deployed, skipping remaining signals")
                break

            decision = self.risk_manager.evaluate_signal(signal)

            if decision["approved"]:
                rejected_streak = 0
                result = await self.executor.execute_trade(decision)
                if result["success"]:
                    self.risk_manager.record_trade({
                        "condition_id": signal.get("condition_id"),
                        "position_size_usdc": decision["position_size_usdc"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    executed += 1
                    src = signal.get("source", "unknown")
                    by_source[src] = by_source.get(src, 0) + 1
                else:
                    log.warning(
                        f"Trade failed: {result.get('reason', 'unknown')} "
                        f"| {signal.get('market_question', '?')[:40]} "
                        f"| token: {signal.get('token_id', '?')[:20]}..."
                    )
            else:
                rejected_streak += 1

        source_str = ", ".join(f"{k}: {v}" for k, v in by_source.items())
        log.info(f"Cycle #{self.cycle_count} complete: {executed} new trades ({source_str})")

    async def run(self, interval_minutes: int = 5, max_cycles: int = 0):
        """Main loop."""
        self.is_running = True
        await self.initialize()

        cycles = 0
        while self.is_running:
            try:
                await self.run_cycle()
                cycles += 1

                if max_cycles > 0 and cycles >= max_cycles:
                    log.info(f"Completed {max_cycles} cycles. Stopping.")
                    break

                self._print_summary()

                from datetime import timedelta
                next_run = datetime.now(timezone.utc) + timedelta(minutes=interval_minutes)
                log.info(f"Next cycle in {interval_minutes} minutes (at {next_run.strftime('%H:%M:%S UTC')})...")
                await asyncio.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                log.info("Keyboard interrupt - shutting down...")
                break
            except Exception as e:
                log.error(f"Cycle error: {e}")
                await asyncio.sleep(60)

        await self.shutdown()

    async def run_once(self):
        """Run a single cycle."""
        await self.initialize()
        await self.run_cycle()
        self._print_summary()
        await self.shutdown()

    def _print_summary(self):
        """Print portfolio and performance (works in both sim and live mode)."""
        mode_label = "SIM" if Config.SIMULATION_MODE else "LIVE"
        perf = self.executor.get_performance()
        strategies = self._enabled_strategies()
        log.info(f"\n{'='*45}")
        log.info(f"[{mode_label}] PORTFOLIO: ${perf['total_value']:.2f} ({perf['return_pct']:+.1f}%)")
        log.info(f"Cash: ${perf['current_balance']:.2f} | Open: {perf['open_positions']} positions")
        log.info(f"Realized P&L: ${perf['total_pnl']:+.2f} | Unrealized: ${perf['unrealized_pnl']:+.2f}")
        log.info(f"Trades: {perf['total_trades']} total | Closed: {perf['closed_trades']} ({perf['wins']}W/{perf['losses']}L)")
        if perf['closed_trades'] > 0:
            log.info(f"Win rate: {perf['win_rate']}% | Avg win: ${perf['avg_win']:+.2f} | Avg loss: ${perf['avg_loss']:+.2f}")

        if perf.get("by_exit_reason"):
            reasons = ", ".join(f"{k}: {v}" for k, v in perf["by_exit_reason"].items())
            log.info(f"Exits by reason: {reasons}")

        if perf.get("by_source"):
            log.info(f"Strategy Performance ({len(strategies)} active):")
            for src, data in perf["by_source"].items():
                wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
                log.info(f"  {src}: {data['trades']} trades, ${data['pnl']:+.2f} PnL, {wr:.0f}% WR")

        if not Config.SIMULATION_MODE and perf.get("daily_loss") is not None:
            log.info(f"Daily P&L: ${perf['daily_loss']:+.2f} (limit: -${Config.MAX_DAILY_LOSS_USDC})")

        log.info(f"{'_'*45}")

    async def shutdown(self):
        """Clean shutdown."""
        log.info("Shutting down...")
        self.is_running = False
        await self.client.close()
        log.info("Shutdown complete.")

    def stop(self):
        self.is_running = False
