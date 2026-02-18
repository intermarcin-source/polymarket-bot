import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from src.core.config import Config
from src.core.polymarket_client import PolymarketClient
from src.utils.logger import setup_logger

log = setup_logger("whale_tracker")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
WHALE_DATA_FILE = DATA_DIR / "whale_wallets.json"
WHALE_POSITIONS_FILE = DATA_DIR / "whale_positions.json"


class WhaleTracker:
    """
    Monitors known profitable wallets on Polymarket.
    Discovers whales via the public leaderboard API.
    Detects when whales open new positions and generates trade signals.
    """

    def __init__(self, client: PolymarketClient):
        self.client = client
        self.whale_wallets: dict[str, dict] = {}
        self.known_positions: dict[str, list[dict]] = {}
        self._load_whale_data()

    def _load_whale_data(self):
        DATA_DIR.mkdir(exist_ok=True)
        if WHALE_DATA_FILE.exists():
            with open(WHALE_DATA_FILE) as f:
                self.whale_wallets = json.load(f)

        if WHALE_POSITIONS_FILE.exists():
            with open(WHALE_POSITIONS_FILE) as f:
                self.known_positions = json.load(f)

    def _save_whale_data(self):
        DATA_DIR.mkdir(exist_ok=True)
        with open(WHALE_DATA_FILE, "w") as f:
            json.dump(self.whale_wallets, f, indent=2)

    def _save_positions(self):
        with open(WHALE_POSITIONS_FILE, "w") as f:
            json.dump(self.known_positions, f, indent=2, default=str)

    def add_whale(self, address: str, label: str = "", notes: str = ""):
        address = address.lower()
        self.whale_wallets[address] = {
            "label": label,
            "notes": notes,
            "added": datetime.now(timezone.utc).isoformat(),
            "total_pnl": 0,
            "win_rate": 0,
            "trades_tracked": 0,
        }
        self._save_whale_data()
        log.info(f"Added whale: {label or address[:10]}")

    async def auto_add_whales(self, max_add: int = 20):
        """Discover profitable wallets from the Polymarket leaderboard."""
        log.info("Discovering whales from Polymarket leaderboard...")
        added = 0

        for period in ["ALL", "MONTH", "WEEK"]:
            if added >= max_add:
                break

            leaderboard = await self.client.get_leaderboard(
                period=period, order_by="PNL", limit=50
            )

            if not leaderboard:
                log.warning(f"Empty leaderboard for period {period}")
                continue

            for entry in leaderboard:
                if added >= max_add:
                    break

                address = entry.get("proxyWallet", entry.get("userAddress", "")).lower()
                if not address or address in self.whale_wallets:
                    continue

                username = entry.get("userName", "")
                pnl = float(entry.get("pnl", 0))
                volume = float(entry.get("vol", entry.get("volume", 0)))
                rank = entry.get("rank", "?")

                if pnl > 0:
                    label = username or f"leaderboard-{address[:8]}"
                    self.whale_wallets[address] = {
                        "label": label,
                        "notes": f"Rank #{rank} ({period}), PnL: ${pnl:,.0f}, Vol: ${volume:,.0f}",
                        "added": datetime.now(timezone.utc).isoformat(),
                        "total_pnl": pnl,
                        "volume": volume,
                        "win_rate": 0,
                        "leaderboard_rank": rank,
                        "period": period,
                    }
                    added += 1
                    log.info(f"Added whale: {label} (rank #{rank}, PnL: ${pnl:,.0f})")

            await asyncio.sleep(0.5)

        # Also top by volume
        volume_leaders = await self.client.get_leaderboard(
            period="MONTH", order_by="VOL", limit=25
        )
        for entry in (volume_leaders or []):
            if added >= max_add:
                break
            address = entry.get("proxyWallet", entry.get("userAddress", "")).lower()
            if not address or address in self.whale_wallets:
                continue
            pnl = float(entry.get("pnl", 0))
            volume = float(entry.get("vol", entry.get("volume", 0)))
            username = entry.get("userName", "")
            if pnl > 0 and volume > 50000:
                label = username or f"vol-whale-{address[:8]}"
                self.whale_wallets[address] = {
                    "label": label,
                    "notes": f"Top volume, PnL: ${pnl:,.0f}, Vol: ${volume:,.0f}",
                    "added": datetime.now(timezone.utc).isoformat(),
                    "total_pnl": pnl,
                    "volume": volume,
                    "win_rate": 0,
                }
                added += 1

        self._save_whale_data()
        log.info(f"Discovered {added} whale wallets from leaderboard (total: {len(self.whale_wallets)})")

    async def scan_for_signals(self) -> list[dict]:
        """Check all whale wallets for new positions."""
        signals = []
        MAX_SIGNALS = 25  # Cap signals per scan to avoid noise

        if not self.whale_wallets:
            log.info("No whale wallets tracked yet.")
            return signals

        log.info(f"Scanning {len(self.whale_wallets)} whale wallets for new activity...")

        for address, meta in self.whale_wallets.items():
            try:
                current_positions = await self.client.get_user_positions(address)
                previous_ids = set()
                for p in self.known_positions.get(address, []):
                    cid = p.get("conditionId", p.get("condition_id", p.get("market", "")))
                    if cid:
                        previous_ids.add(cid)

                for pos in current_positions:
                    if len(signals) >= MAX_SIGNALS:
                        break

                    cond_id = pos.get("conditionId", pos.get("condition_id", pos.get("market", "")))
                    size = float(pos.get("size", pos.get("tokens", 0)))
                    price = float(pos.get("avgPrice", pos.get("averagePrice", 0)))
                    value = size * price if price > 0 else float(pos.get("initialValue", 0))

                    if cond_id and cond_id not in previous_ids and value >= Config.MIN_WHALE_FOLLOW_SIZE:
                        signal = {
                            "source": "whale_tracker",
                            "type": "whale_new_position",
                            "whale_address": address,
                            "whale_label": meta.get("label", address[:10]),
                            "whale_pnl": meta.get("total_pnl", 0),
                            "condition_id": cond_id,
                            "market_question": pos.get("title", pos.get("question", f"Whale position: {pos.get('outcome', '?')}")),
                            "recommended_outcome": pos.get("outcome", pos.get("title", "")),
                            "outcome": pos.get("outcome", pos.get("title", "")),
                            "token_id": pos.get("tokenId", pos.get("token_id", "")),
                            "market_price": price if price > 0 else None,
                            "whale_size": size,
                            "whale_avg_price": price,
                            "whale_value_usdc": value,
                            "confidence": 0.65,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        signals.append(signal)
                        log.info(
                            f"[SIGNAL] Whale {meta.get('label', address[:10])} "
                            f"new position: {pos.get('outcome', '?')} (${value:.0f})"
                        )

                self.known_positions[address] = current_positions
                await asyncio.sleep(0.3)

            except Exception as e:
                log.error(f"Error scanning whale {address}: {e}")

        self._save_positions()
        log.info(f"Whale scan complete: {len(signals)} signals generated")
        return signals
