"""
Position manager for tracking and managing open positions.

Coordinates between:
- Paper trader (execution)
- Data manager (settlement data)
- Edge calculator (position evaluation)
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Optional

from config.settings import MAX_POSITION_SIZE
from data.collectors import DataManager
from trading.paper_trader import PaperTrader, PaperTrade
from utils.logger import get_logger
from utils.helpers import format_currency, calculate_days_until

logger = get_logger("position_manager")


class PositionManager:
    """
    Manages open positions and handles settlement.

    Responsibilities:
    - Track all open positions
    - Check for positions needing settlement
    - Calculate current exposure
    - Provide position reports
    """

    def __init__(
        self,
        paper_trader: PaperTrader = None,
        data_manager: DataManager = None
    ):
        """
        Initialize the position manager.

        Args:
            paper_trader: Paper trader instance
            data_manager: Data manager instance
        """
        self.paper_trader = paper_trader or PaperTrader()
        self.data_manager = data_manager or DataManager()

        logger.info("Position manager initialized")

    def get_open_positions(self) -> List[PaperTrade]:
        """Get all open positions."""
        return self.paper_trader.get_open_positions()

    def get_position_summary(self) -> Dict:
        """
        Get summary of current positions.

        Returns:
            Dictionary with position statistics
        """
        positions = self.get_open_positions()

        total_exposure = 0
        by_location = {}
        by_confidence = {"high": 0, "medium": 0, "low": 0}
        by_direction = {"BUY_YES": 0, "BUY_NO": 0}

        for pos in positions:
            exposure = pos.entry_price * pos.contracts
            total_exposure += exposure

            # By location
            if pos.location not in by_location:
                by_location[pos.location] = {"count": 0, "exposure": 0}
            by_location[pos.location]["count"] += 1
            by_location[pos.location]["exposure"] += exposure

            # By confidence
            if pos.confidence in by_confidence:
                by_confidence[pos.confidence] += 1

            # By direction
            if pos.direction in by_direction:
                by_direction[pos.direction] += 1

        return {
            "total_positions": len(positions),
            "total_exposure": total_exposure,
            "available_capital": MAX_POSITION_SIZE - total_exposure,
            "by_location": by_location,
            "by_confidence": by_confidence,
            "by_direction": by_direction,
            "positions": [self._position_to_dict(p) for p in positions]
        }

    def _position_to_dict(self, position: PaperTrade) -> Dict:
        """Convert position to dictionary for reporting."""
        days_out = calculate_days_until(position.target_date)

        return {
            "id": position.id,
            "ticker": position.ticker,
            "direction": position.direction,
            "contracts": position.contracts,
            "entry_price": position.entry_price,
            "exposure": position.entry_price * position.contracts,
            "edge_at_entry": position.edge_at_entry,
            "confidence": position.confidence,
            "target_date": position.target_date.isoformat(),
            "days_until_settlement": days_out,
            "temp_threshold": position.temp_threshold
        }

    def check_settlements(self) -> List[Dict]:
        """
        Check for positions that can be settled.

        Checks positions where:
        - Target date has passed
        - Actual weather data is available

        Returns:
            List of settlement results
        """
        positions = self.get_open_positions()
        results = []
        today = date.today()

        for position in positions:
            # Only settle if target date has passed
            if position.target_date >= today:
                continue

            # Try to get actual weather
            actual = self.data_manager.get_actual(
                position.location,
                position.target_date
            )

            if not actual:
                logger.warning(
                    f"No actual data for {position.ticker} ({position.target_date})"
                )
                continue

            # Determine outcome
            if position.direction.endswith("YES"):
                # We bought YES - check if condition was met
                if "HIGH" in position.ticker.upper():
                    actual_temp = actual.high_temp_f
                    condition_met = actual_temp > position.temp_threshold
                else:  # LOW
                    actual_temp = actual.low_temp_f
                    condition_met = actual_temp < position.temp_threshold
            else:
                # We bought NO - inverse
                if "HIGH" in position.ticker.upper():
                    actual_temp = actual.high_temp_f
                    condition_met = actual_temp <= position.temp_threshold
                else:
                    actual_temp = actual.low_temp_f
                    condition_met = actual_temp >= position.temp_threshold

            outcome = "YES" if condition_met else "NO"

            # Settle the position
            try:
                settled = self.paper_trader.settle_position(
                    position.id,
                    outcome,
                    actual_temp
                )

                results.append({
                    "trade_id": position.id,
                    "ticker": position.ticker,
                    "outcome": outcome,
                    "actual_temp": actual_temp,
                    "threshold": position.temp_threshold,
                    "our_prediction": "YES" if position.direction == "BUY_YES" else "NO",
                    "correct": (outcome == "YES" and position.direction == "BUY_YES") or
                              (outcome == "NO" and position.direction == "BUY_NO"),
                    "net_pnl": settled.net_pnl
                })

                logger.info(
                    f"Settled {position.ticker}: {outcome} @ {actual_temp}°F, "
                    f"P&L: {format_currency(settled.net_pnl)}"
                )

            except Exception as e:
                logger.error(f"Error settling {position.id}: {e}")

        return results

    def get_expiring_positions(self, days: int = 1) -> List[PaperTrade]:
        """
        Get positions expiring within specified days.

        Args:
            days: Number of days to look ahead

        Returns:
            List of positions expiring soon
        """
        positions = self.get_open_positions()
        cutoff = date.today() + timedelta(days=days)

        return [p for p in positions if p.target_date <= cutoff]

    def can_open_position(
        self,
        ticker: str,
        contracts: int,
        entry_price: float
    ) -> Dict:
        """
        Check if a new position can be opened.

        Args:
            ticker: Market ticker
            contracts: Number of contracts
            entry_price: Entry price per contract

        Returns:
            Dictionary with 'allowed' bool and 'reason' string
        """
        # Check for existing position
        existing = self.paper_trader.get_position(ticker)
        if existing:
            return {
                "allowed": False,
                "reason": f"Position already exists in {ticker}"
            }

        # Check exposure limit
        summary = self.get_position_summary()
        new_exposure = contracts * entry_price
        total_after = summary["total_exposure"] + new_exposure

        if total_after > MAX_POSITION_SIZE:
            return {
                "allowed": False,
                "reason": f"Would exceed max exposure. Available: {format_currency(summary['available_capital'])}"
            }

        return {
            "allowed": True,
            "reason": "OK",
            "new_exposure": new_exposure,
            "total_after": total_after
        }

    def get_exposure_report(self) -> str:
        """Generate a text report of current exposure."""
        summary = self.get_position_summary()

        lines = [
            "=" * 50,
            "POSITION EXPOSURE REPORT",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 50,
            "",
            f"Total Positions: {summary['total_positions']}",
            f"Total Exposure: {format_currency(summary['total_exposure'])}",
            f"Available Capital: {format_currency(summary['available_capital'])}",
            "",
            "By Location:"
        ]

        for loc, data in summary["by_location"].items():
            lines.append(f"  {loc}: {data['count']} positions, {format_currency(data['exposure'])}")

        lines.extend([
            "",
            "By Confidence:",
            f"  High: {summary['by_confidence']['high']}",
            f"  Medium: {summary['by_confidence']['medium']}",
            f"  Low: {summary['by_confidence']['low']}",
            "",
            "By Direction:",
            f"  BUY YES: {summary['by_direction']['BUY_YES']}",
            f"  BUY NO: {summary['by_direction']['BUY_NO']}",
            "",
            "Open Positions:"
        ])

        for pos in summary["positions"]:
            lines.append(
                f"  {pos['ticker']}: {pos['direction']} x{pos['contracts']} @ "
                f"{format_currency(pos['entry_price'])} | Edge: {pos['edge_at_entry']:.1%} | "
                f"{pos['days_until_settlement']}d to settlement"
            )

        lines.append("=" * 50)

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    print("Testing PositionManager...")

    manager = PositionManager()

    # Get position summary
    summary = manager.get_position_summary()
    print(f"\nPosition Summary:")
    print(f"  Total positions: {summary['total_positions']}")
    print(f"  Total exposure: {format_currency(summary['total_exposure'])}")
    print(f"  Available: {format_currency(summary['available_capital'])}")

    # Generate report
    print("\n" + manager.get_exposure_report())

    # Check if we can open a new position
    can_open = manager.can_open_position("HIGHNY-25JAN20-T50", 10, 0.50)
    print(f"\nCan open new position: {can_open['allowed']}")
    if not can_open["allowed"]:
        print(f"  Reason: {can_open['reason']}")
