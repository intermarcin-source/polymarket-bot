import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from src.core.config import Config
from src.core.polymarket_client import PolymarketClient
from src.utils.logger import setup_logger

log = setup_logger("bot_scanner")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
BOT_DATA_FILE = DATA_DIR / "tracked_bots.json"


class BotScanner:
    """
    Identifies automated trading accounts on Polymarket with high profitability.
    Bot detection heuristics:
    - High trade frequency (many trades in short time windows)
    - Consistent position sizing
    - Activity across many markets simultaneously
    - Round-number bet amounts
    - Fast reaction to market events
    """

    def __init__(self, client: PolymarketClient):
        self.client = client
        self.tracked_bots: dict[str, dict] = {}
        self.bot_positions: dict[str, list[dict]] = {}
        self._load_data()

    def _load_data(self):
        DATA_DIR.mkdir(exist_ok=True)
        if BOT_DATA_FILE.exists():
            with open(BOT_DATA_FILE) as f:
                self.tracked_bots = json.load(f)

    def _save_data(self):
        DATA_DIR.mkdir(exist_ok=True)
        with open(BOT_DATA_FILE, "w") as f:
            json.dump(self.tracked_bots, f, indent=2, default=str)

    def _is_likely_bot(self, trades: list[dict], address: str) -> tuple[bool, float]:
        """
        Heuristic analysis to determine if a wallet is likely an automated bot.
        Returns (is_bot, bot_score).
        """
        if len(trades) < 5:
            return False, 0.0

        score = 0.0

        # 1. Trade frequency - bots trade much more frequently
        timestamps = []
        for t in trades:
            ts = t.get("timestamp", t.get("createdAt", ""))
            if ts:
                try:
                    if isinstance(ts, (int, float)):
                        timestamps.append(ts)
                    else:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        timestamps.append(dt.timestamp())
                except (ValueError, TypeError):
                    pass

        if len(timestamps) >= 2:
            timestamps.sort()
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals) if intervals else float("inf")
            if avg_interval < 60:  # trades less than 1 min apart
                score += 0.3
            elif avg_interval < 300:  # less than 5 min
                score += 0.2
            elif avg_interval < 900:  # less than 15 min
                score += 0.1

        # 2. Consistent position sizes (bots often use fixed amounts)
        sizes = [float(t.get("size", 0)) for t in trades if float(t.get("size", 0)) > 0]
        if len(sizes) >= 3:
            avg_size = sum(sizes) / len(sizes)
            if avg_size > 0:
                variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
                cv = (variance ** 0.5) / avg_size  # coefficient of variation
                if cv < 0.1:  # very consistent sizing
                    score += 0.25
                elif cv < 0.3:
                    score += 0.15

        # 3. Round number bet amounts
        round_count = sum(1 for s in sizes if s == int(s) or s % 5 == 0)
        if sizes and round_count / len(sizes) > 0.7:
            score += 0.15

        # 4. High number of distinct markets
        markets = {t.get("market", t.get("conditionId", "")) for t in trades}
        if len(markets) > 10:
            score += 0.2
        elif len(markets) > 5:
            score += 0.1

        is_bot = score >= 0.45
        return is_bot, min(score, 1.0)

    async def evaluate_and_track_bots(self, max_add: int = 10):
        """
        Discover high-volume traders from leaderboard, detect bot-like behavior
        via their activity patterns, and track profitable ones.
        """
        log.info("Discovering potential bots from leaderboard high-volume traders...")
        added = 0

        # Get high-volume traders - bots tend to have high volume
        leaderboard = await self.client.get_leaderboard(
            period="MONTH", order_by="VOL", limit=50
        )

        if not leaderboard:
            log.warning("Could not fetch leaderboard for bot discovery")
            self._save_data()
            return

        for entry in leaderboard:
            if added >= max_add:
                break

            address = entry.get("proxyWallet", entry.get("userAddress", "")).lower()
            if not address or address in self.tracked_bots:
                continue

            pnl = float(entry.get("pnl", 0))
            volume = float(entry.get("vol", entry.get("volume", 0)))
            username = entry.get("userName", "")

            if pnl <= 0:
                continue

            # Check activity patterns for bot-like behavior
            activity = await self.client.get_user_activity(address, limit=200)
            if len(activity) < 10:
                await asyncio.sleep(0.3)
                continue

            is_bot, bot_score = self._is_likely_bot(activity, address)
            if is_bot and pnl > 0:
                label = f"bot-{username}" if username else f"bot-{address[:8]}"
                self.tracked_bots[address] = {
                    "label": label,
                    "bot_score": bot_score,
                    "win_rate": 0,
                    "estimated_pnl": pnl,
                    "volume": volume,
                    "total_positions": len(activity),
                    "discovered": datetime.now(timezone.utc).isoformat(),
                }
                added += 1
                log.info(
                    f"Tracking bot {label} "
                    f"(score: {bot_score:.2f}, PnL: ${pnl:,.0f}, Vol: ${volume:,.0f})"
                )

            await asyncio.sleep(0.3)

        self._save_data()
        log.info(f"Now tracking {len(self.tracked_bots)} bots total")

    async def _evaluate_performance(self, address: str) -> Optional[dict]:
        """Check a bot's trading performance."""
        try:
            positions = await self.client.get_user_positions(address)
            if not positions:
                return None

            wins = 0
            losses = 0
            total_pnl = 0.0

            for pos in positions:
                size = float(pos.get("size", 0))
                avg_price = float(pos.get("avgPrice", 0))
                current_price = float(pos.get("currentPrice", avg_price))

                if size > 0 and avg_price > 0:
                    pnl = (current_price - avg_price) * size
                    total_pnl += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

            total = wins + losses
            return {
                "win_rate": wins / total if total > 0 else 0,
                "wins": wins,
                "losses": losses,
                "estimated_pnl": total_pnl,
                "total_positions": len(positions),
            }
        except Exception as e:
            log.error(f"Failed to evaluate bot {address}: {e}")
            return None

    async def scan_for_signals(self) -> list[dict]:
        """
        Check tracked bots for new positions and generate signals.
        """
        signals = []
        MAX_SIGNALS = 25  # Cap signals per scan to avoid noise
        log.info(f"Scanning {len(self.tracked_bots)} tracked bots for activity...")

        for address, meta in self.tracked_bots.items():
            try:
                current_positions = await self.client.get_user_positions(address)
                previous_ids = {
                    p.get("conditionId", "") for p in self.bot_positions.get(address, [])
                }

                for pos in current_positions:
                    if len(signals) >= MAX_SIGNALS:
                        break

                    cond_id = pos.get("conditionId", "")
                    size = float(pos.get("size", 0))
                    price = float(pos.get("avgPrice", 0))
                    value = size * price

                    if cond_id and cond_id not in previous_ids and value > 0:
                        signal = {
                            "source": "bot_scanner",
                            "type": "bot_new_position",
                            "bot_address": address,
                            "bot_label": meta.get("label", address[:10]),
                            "bot_score": meta.get("bot_score", 0),
                            "bot_win_rate": meta.get("win_rate", 0),
                            "condition_id": cond_id,
                            "market_question": pos.get("title", pos.get("question", f"Bot position: {pos.get('outcome', '?')}")),
                            "recommended_outcome": pos.get("outcome", ""),
                            "outcome": pos.get("outcome", ""),
                            "token_id": pos.get("tokenId", ""),
                            "market_price": price if price > 0 else None,
                            "bot_size": size,
                            "bot_avg_price": price,
                            "bot_value_usdc": value,
                            "confidence": min(
                                0.4
                                + meta.get("win_rate", 0) * 0.3
                                + meta.get("bot_score", 0) * 0.2,
                                0.9,
                            ),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        signals.append(signal)
                        log.info(
                            f"[SIGNAL] Bot {meta.get('label', address[:10])} "
                            f"new position: {pos.get('outcome', '?')} (${value:.0f})"
                        )

                self.bot_positions[address] = current_positions
                await asyncio.sleep(0.5)

            except Exception as e:
                log.error(f"Error scanning bot {address}: {e}")

        log.info(f"Bot scan complete: {len(signals)} signals generated")
        return signals
