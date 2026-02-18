"""
Kill Switch Utilities
=====================
Emergency halt for the trading bot.
Create the kill switch file to immediately stop all trading.
Delete it to resume.

Usage from Python:
    from src.core.kill_switch import activate_kill_switch, deactivate_kill_switch
    activate_kill_switch("Market crash detected")
    deactivate_kill_switch()

Usage from CLI (on server):
    bash deploy/kill_switch.sh on "reason"
    bash deploy/kill_switch.sh off
"""

from pathlib import Path
from datetime import datetime, timezone
from src.core.config import Config


def activate_kill_switch(reason: str = "Manual activation"):
    """Create kill switch file to immediately halt all trading."""
    path = Path(Config.KILL_SWITCH_FILE)
    path.parent.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    path.write_text(f"KILL SWITCH ACTIVATED: {reason} | {timestamp}")
    return True


def deactivate_kill_switch():
    """Remove kill switch file to resume trading."""
    path = Path(Config.KILL_SWITCH_FILE)
    if path.exists():
        path.unlink()
        return True
    return False


def is_active() -> bool:
    """Check if kill switch is currently active."""
    return Path(Config.KILL_SWITCH_FILE).exists()


def get_status() -> dict:
    """Get kill switch status and details."""
    path = Path(Config.KILL_SWITCH_FILE)
    if path.exists():
        return {
            "active": True,
            "details": path.read_text().strip(),
        }
    return {"active": False, "details": ""}
