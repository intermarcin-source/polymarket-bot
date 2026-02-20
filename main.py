"""
BTC 5-Minute Sniper Bot
========================
Trades Polymarket BTC Up/Down 5-minute markets.
Late-window snipe strategy: enters at T-10s when direction is ~90% locked in.

Usage:
    python main.py              # Run continuously
    python main.py --once       # Trade next window only
    python main.py --sim        # Force simulation mode
"""

import sys
import asyncio
import argparse

if sys.platform == "win32" and sys.version_info < (3, 13):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.orchestrator import TradingOrchestrator
from src.core.config import Config
from src.utils.logger import setup_logger

log = setup_logger("main")


async def main():
    parser = argparse.ArgumentParser(description="BTC 5-Minute Sniper Bot")
    parser.add_argument("--once", action="store_true", help="Trade next window and exit")
    parser.add_argument("--sim", action="store_true", help="Force simulation mode")
    parser.add_argument("--interval", type=int, default=0, help="(ignored, kept for compatibility)")
    parser.add_argument("--max-cycles", type=int, default=0, help="Max polling cycles (0=infinite)")
    args = parser.parse_args()

    if args.sim:
        Config.SIMULATION_MODE = True

    errors = Config.validate()
    if errors:
        log.error("Configuration errors:")
        for e in errors:
            log.error(f"  - {e}")
        if not Config.SIMULATION_MODE:
            return

    # Run with crash recovery
    while True:
        try:
            bot = TradingOrchestrator()
            if args.once:
                await bot.run_once()
                break
            else:
                await bot.run(max_cycles=args.max_cycles)
                break
        except KeyboardInterrupt:
            log.info("Keyboard interrupt - exiting.")
            break
        except Exception as e:
            log.error(f"Bot crashed: {e}", exc_info=True)
            if args.once:
                break
            log.info("Restarting in 30 seconds...")
            await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(main())
