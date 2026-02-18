"""
Polymarket Multi-Strategy Trading Bot
======================================
Multi-strategy prediction market trading bot:
1. Whale copy trading - mirrors profitable wallets
2. Single-market arbitrage - YES + NO < $1.00 guaranteed profit
3. Market making - bid-ask spread capture on high-volume markets
4. Mean reversion - trade overreaction bounces
5. NegRisk rebalancing - multi-outcome probability sum arbitrage
6. Bot copy trading - copies high-performing automated traders

Usage:
    python main.py              # Run bot (continuous mode, 5 min cycles)
    python main.py --once       # Run a single cycle
    python main.py --dashboard  # Show performance dashboard
    python main.py --reset      # Reset simulation data
    python main.py --interval 10  # Set cycle interval in minutes
"""

import sys
import asyncio
import argparse

# Fix for Windows asyncio (use selector on older Python, not needed on 3.13+)
if sys.platform == "win32" and sys.version_info < (3, 13):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.orchestrator import TradingOrchestrator
from src.core.config import Config
from src.dashboard import show_dashboard
from src.core.simulator import PaperTrader
from src.core.polymarket_client import PolymarketClient
from src.utils.logger import setup_logger

log = setup_logger("main")


async def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading Bot")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--dashboard", action="store_true", help="Show performance dashboard")
    parser.add_argument("--reset", action="store_true", help="Reset simulation data")
    parser.add_argument("--interval", type=int, default=5, help="Cycle interval in minutes (default: 5)")
    parser.add_argument("--max-cycles", type=int, default=0, help="Max cycles to run (0 = infinite)")
    args = parser.parse_args()

    if args.dashboard:
        show_dashboard()
        return

    if args.reset:
        client = PolymarketClient()
        sim = PaperTrader(client)
        sim.reset()
        log.info("Simulation data reset successfully!")
        await client.close()
        return

    # Validate config
    errors = Config.validate()
    if errors:
        log.error("Configuration errors found:")
        for e in errors:
            log.error(f"  - {e}")
        log.error("Please check your .env file. See .env.example for required values.")
        if not Config.SIMULATION_MODE:
            return

    # Run the bot with crash recovery
    while True:
        try:
            bot = TradingOrchestrator()
            if args.once:
                await bot.run_once()
                break
            else:
                await bot.run(interval_minutes=args.interval, max_cycles=args.max_cycles)
                break
        except KeyboardInterrupt:
            log.info("Keyboard interrupt - exiting.")
            break
        except Exception as e:
            log.error(f"Bot crashed: {e}", exc_info=True)
            if args.once:
                break
            log.info("Restarting in 60 seconds...")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
