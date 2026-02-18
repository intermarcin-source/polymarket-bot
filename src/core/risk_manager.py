from datetime import datetime, timezone
from src.core.config import Config
from src.utils.logger import setup_logger

log = setup_logger("risk_manager")


class RiskManager:
    """
    Controls position sizing, portfolio exposure, and trade approval.
    Every trade signal must pass through the risk manager before execution.

    Position sizing scales with portfolio value (compounding):
    - Max bet per trade: 5% of portfolio value
    - Max total exposure: 75% of portfolio value
    - As profits grow the balance, bets get bigger automatically
    """

    def __init__(self):
        self.open_positions: list[dict] = []  # tracks what we hold
        self.daily_trades: list[dict] = []
        self.daily_pnl: float = 0.0
        self.total_exposure: float = 0.0
        self.portfolio_value: float = 2000.0  # updated each cycle by orchestrator

    @property
    def max_bet_size(self) -> float:
        """5% of current portfolio value — scales as we grow."""
        return self.portfolio_value * 0.05

    @property
    def max_exposure(self) -> float:
        """75% of current portfolio value — scales as we grow."""
        return self.portfolio_value * 0.75

    @property
    def available_capital(self) -> float:
        return max(0, self.max_exposure - self.total_exposure)

    def evaluate_signal(self, signal: dict) -> dict:
        """
        Evaluate a trade signal and return an approved/rejected decision
        with position sizing.
        """
        confidence = signal.get("confidence", 0)
        source = signal.get("source", "unknown")
        condition_id = signal.get("condition_id", "")

        decision = {
            "approved": False,
            "signal": signal,
            "reason": "",
            "position_size_usdc": 0.0,
            "max_loss_usdc": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Rule 1: Minimum confidence
        if confidence < 0.5:
            decision["reason"] = f"Confidence {confidence:.2f} below minimum 0.50"
            log.info(f"[RISK] REJECTED: {decision['reason']}")
            return decision

        # Rule 2: Check portfolio exposure
        if self.total_exposure >= self.max_exposure:
            decision["reason"] = f"Portfolio exposure ${self.total_exposure:.0f} at max ${self.max_exposure:.0f}"
            log.debug(f"[RISK] REJECTED: {decision['reason']}")
            return decision

        # Rule 3: No duplicate positions in same market
        for pos in self.open_positions:
            if pos.get("condition_id") == condition_id:
                decision["reason"] = f"Already have position in this market"
                log.info(f"[RISK] REJECTED: {decision['reason']}")
                return decision

        # Rule 4: Daily trade limit (max 1000 trades per day)
        today = datetime.now(timezone.utc).date()
        todays_trades = [
            t for t in self.daily_trades
            if datetime.fromisoformat(t["timestamp"]).date() == today
        ]
        if len(todays_trades) >= 1000:
            decision["reason"] = "Daily trade limit reached (1000)"
            log.info(f"[RISK] REJECTED: {decision['reason']}")
            return decision

        # Rule 5: Max concentration (no single position > 30% of exposure cap)
        max_single = self.max_exposure * 0.30

        # Calculate position size based on confidence and edge
        base_size = self._calculate_position_size(signal)
        position_size = min(base_size, self.max_bet_size, self.available_capital, max_single)

        if position_size < 1.0:  # minimum $1
            decision["reason"] = f"Position size too small: ${position_size:.2f}"
            log.info(f"[RISK] REJECTED: {decision['reason']}")
            return decision

        # Calculate max loss
        market_price = signal.get("market_price", 0.5)
        max_loss = position_size * market_price  # worst case: outcome goes to 0

        # Approved
        decision["approved"] = True
        decision["position_size_usdc"] = round(position_size, 2)
        decision["max_loss_usdc"] = round(max_loss, 2)
        decision["reason"] = "Passed all risk checks"

        log.info(
            f"[RISK] APPROVED: ${position_size:.2f} "
            f"(conf: {confidence:.2f}, source: {source}, max_loss: ${max_loss:.2f})"
        )
        return decision

    def _calculate_position_size(self, signal: dict) -> float:
        """
        Kelly-inspired position sizing based on confidence and edge.
        Higher confidence + larger edge = bigger position.
        Scales with portfolio value for compounding growth.
        """
        confidence = signal.get("confidence", 0)
        edge = abs(signal.get("avg_edge", signal.get("edge", 0)))

        # Base sizing: fraction of dynamic max bet based on confidence
        base = self.max_bet_size * confidence

        # Edge bonus: larger edge -> slightly larger position
        edge_bonus = 1.0 + min(edge * 2, 0.5)  # up to 50% bonus for large edges

        return base * edge_bonus

    def record_trade(self, trade: dict):
        """Record an executed trade."""
        self.daily_trades.append(trade)
        self.open_positions.append(trade)
        self.total_exposure += trade.get("position_size_usdc", 0)
        log.info(f"Recorded trade. Total exposure: ${self.total_exposure:.2f}")

    def close_position(self, condition_id: str, pnl: float):
        """Close a position and update tracking."""
        self.open_positions = [
            p for p in self.open_positions if p.get("condition_id") != condition_id
        ]
        self.daily_pnl += pnl
        # Recalculate exposure
        self.total_exposure = sum(
            p.get("position_size_usdc", 0) for p in self.open_positions
        )

    def get_portfolio_summary(self) -> dict:
        return {
            "open_positions": len(self.open_positions),
            "total_exposure_usdc": round(self.total_exposure, 2),
            "available_capital_usdc": round(self.available_capital, 2),
            "daily_pnl_usdc": round(self.daily_pnl, 2),
            "daily_trades": len(self.daily_trades),
            "portfolio_value_usdc": round(self.portfolio_value, 2),
            "max_exposure_usdc": round(self.max_exposure, 2),
            "max_bet_usdc": round(self.max_bet_size, 2),
        }
