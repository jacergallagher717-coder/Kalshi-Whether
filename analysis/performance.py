"""
Performance analytics for paper trading results.

Key Metrics:
1. Overall P&L (gross and net)
2. Win rate by various dimensions
3. Edge analysis (predicted vs realized)
4. Risk metrics
"""

import sqlite3
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from config.settings import TRADES_DB
from utils.logger import get_logger
from utils.helpers import format_currency, format_percent

logger = get_logger("performance")


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    gross_pnl: float
    total_fees: float
    net_pnl: float
    avg_edge_at_entry: float
    avg_pnl_per_trade: float
    largest_win: float
    largest_loss: float
    max_drawdown: float
    sharpe_ratio: Optional[float]
    profit_factor: Optional[float]


class PerformanceAnalyzer:
    """
    Analyzes trading performance from paper trading results.

    Provides:
    - Overall statistics
    - Performance breakdowns by various dimensions
    - Risk metrics
    - Trend analysis
    """

    def __init__(self, db_path: str = None):
        """Initialize the analyzer."""
        self.db_path = db_path or TRADES_DB
        logger.info("Performance analyzer initialized")

    def get_overall_metrics(self) -> PerformanceMetrics:
        """
        Calculate overall performance metrics.

        Returns:
            PerformanceMetrics with all key statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all settled trades
        cursor.execute('''
            SELECT
                net_pnl, gross_pnl, simulated_fees, edge_at_entry,
                direction, confidence, created_at
            FROM paper_trades
            WHERE status = 'SETTLED'
            ORDER BY created_at ASC
        ''')

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return PerformanceMetrics(
                total_trades=0, wins=0, losses=0, win_rate=0,
                gross_pnl=0, total_fees=0, net_pnl=0,
                avg_edge_at_entry=0, avg_pnl_per_trade=0,
                largest_win=0, largest_loss=0, max_drawdown=0,
                sharpe_ratio=None, profit_factor=None
            )

        # Extract data
        net_pnls = [row[0] for row in rows]
        gross_pnls = [row[1] for row in rows]
        fees = [row[2] for row in rows]
        edges = [row[3] for row in rows]

        # Basic counts
        total = len(rows)
        wins = sum(1 for pnl in net_pnls if pnl > 0)
        losses = total - wins

        # P&L metrics
        gross_pnl = sum(gross_pnls)
        total_fees = sum(fees)
        net_pnl = sum(net_pnls)
        avg_pnl = net_pnl / total if total > 0 else 0

        # Win/loss extremes
        largest_win = max(net_pnls) if net_pnls else 0
        largest_loss = min(net_pnls) if net_pnls else 0

        # Edge analysis
        avg_edge = np.mean(edges) if edges else 0

        # Calculate max drawdown
        cumulative = np.cumsum(net_pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = max(drawdowns) if len(drawdowns) > 0 else 0

        # Sharpe ratio (annualized, assuming daily trades)
        if len(net_pnls) > 1 and np.std(net_pnls) > 0:
            daily_return = np.mean(net_pnls)
            daily_std = np.std(net_pnls)
            sharpe = (daily_return / daily_std) * np.sqrt(252)  # Annualized
        else:
            sharpe = None

        # Profit factor (gross wins / gross losses)
        gross_wins = sum(pnl for pnl in gross_pnls if pnl > 0)
        gross_losses = abs(sum(pnl for pnl in gross_pnls if pnl < 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else None

        return PerformanceMetrics(
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=wins / total if total > 0 else 0,
            gross_pnl=gross_pnl,
            total_fees=total_fees,
            net_pnl=net_pnl,
            avg_edge_at_entry=avg_edge,
            avg_pnl_per_trade=avg_pnl,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor
        )

    def get_performance_by_confidence(self) -> Dict[str, Dict]:
        """Get performance breakdown by confidence level."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                confidence,
                COUNT(*) as trades,
                SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(net_pnl) as net_pnl,
                AVG(edge_at_entry) as avg_edge
            FROM paper_trades
            WHERE status = 'SETTLED'
            GROUP BY confidence
        ''')

        results = {}
        for row in cursor.fetchall():
            conf = row[0]
            trades = row[1]
            wins = row[2]
            results[conf] = {
                "trades": trades,
                "wins": wins,
                "losses": trades - wins,
                "win_rate": wins / trades if trades > 0 else 0,
                "net_pnl": row[3],
                "avg_edge": row[4]
            }

        conn.close()
        return results

    def get_performance_by_edge_bucket(self) -> Dict[str, Dict]:
        """Get performance breakdown by edge bucket."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                CASE
                    WHEN ABS(edge_at_entry) >= 0.25 THEN '25%+'
                    WHEN ABS(edge_at_entry) >= 0.20 THEN '20-25%'
                    WHEN ABS(edge_at_entry) >= 0.15 THEN '15-20%'
                    WHEN ABS(edge_at_entry) >= 0.10 THEN '10-15%'
                    ELSE '<10%'
                END as edge_bucket,
                COUNT(*) as trades,
                SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(net_pnl) as net_pnl,
                AVG(edge_at_entry) as avg_edge
            FROM paper_trades
            WHERE status = 'SETTLED'
            GROUP BY edge_bucket
            ORDER BY avg_edge DESC
        ''')

        results = {}
        for row in cursor.fetchall():
            bucket = row[0]
            trades = row[1]
            wins = row[2]
            results[bucket] = {
                "trades": trades,
                "wins": wins,
                "win_rate": wins / trades if trades > 0 else 0,
                "net_pnl": row[3],
                "avg_edge": row[4]
            }

        conn.close()
        return results

    def get_performance_by_days_out(self) -> Dict[int, Dict]:
        """Get performance breakdown by forecast horizon."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                CAST(julianday(target_date) - julianday(date(created_at)) AS INTEGER) as days_out,
                COUNT(*) as trades,
                SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(net_pnl) as net_pnl,
                AVG(edge_at_entry) as avg_edge
            FROM paper_trades
            WHERE status = 'SETTLED'
            GROUP BY days_out
            ORDER BY days_out
        ''')

        results = {}
        for row in cursor.fetchall():
            days = int(row[0])
            trades = row[1]
            wins = row[2]
            results[days] = {
                "trades": trades,
                "wins": wins,
                "win_rate": wins / trades if trades > 0 else 0,
                "net_pnl": row[3],
                "avg_edge": row[4]
            }

        conn.close()
        return results

    def get_performance_by_location(self) -> Dict[str, Dict]:
        """Get performance breakdown by location."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                location,
                COUNT(*) as trades,
                SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(net_pnl) as net_pnl,
                AVG(edge_at_entry) as avg_edge
            FROM paper_trades
            WHERE status = 'SETTLED'
            GROUP BY location
        ''')

        results = {}
        for row in cursor.fetchall():
            loc = row[0]
            trades = row[1]
            wins = row[2]
            results[loc] = {
                "trades": trades,
                "wins": wins,
                "win_rate": wins / trades if trades > 0 else 0,
                "net_pnl": row[3],
                "avg_edge": row[4]
            }

        conn.close()
        return results

    def get_daily_pnl(self, days: int = 30) -> List[Dict]:
        """Get daily P&L for the last N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_date = (date.today() - timedelta(days=days)).isoformat()

        cursor.execute('''
            SELECT
                date(settled_at) as settle_date,
                COUNT(*) as trades,
                SUM(net_pnl) as net_pnl,
                SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins
            FROM paper_trades
            WHERE status = 'SETTLED' AND settled_at >= ?
            GROUP BY settle_date
            ORDER BY settle_date
        ''', (start_date,))

        results = []
        for row in cursor.fetchall():
            results.append({
                "date": row[0],
                "trades": row[1],
                "net_pnl": row[2],
                "wins": row[3],
                "win_rate": row[3] / row[1] if row[1] > 0 else 0
            })

        conn.close()
        return results

    def get_cumulative_pnl(self) -> List[Tuple[datetime, float]]:
        """Get cumulative P&L over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT settled_at, net_pnl
            FROM paper_trades
            WHERE status = 'SETTLED'
            ORDER BY settled_at ASC
        ''')

        results = []
        cumulative = 0
        for row in cursor.fetchall():
            cumulative += row[1]
            results.append((row[0], cumulative))

        conn.close()
        return results

    def get_model_accuracy(self) -> Dict[str, float]:
        """
        Calculate how accurate our probability estimates were.

        Compares our predicted probability to actual outcomes.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                our_probability,
                direction,
                settlement_outcome
            FROM paper_trades
            WHERE status = 'SETTLED'
        ''')

        # Group predictions by probability bucket
        buckets = {
            "0-20%": {"predictions": 0, "correct": 0},
            "20-40%": {"predictions": 0, "correct": 0},
            "40-60%": {"predictions": 0, "correct": 0},
            "60-80%": {"predictions": 0, "correct": 0},
            "80-100%": {"predictions": 0, "correct": 0}
        }

        for row in cursor.fetchall():
            prob = row[0]
            direction = row[1]
            outcome = row[2]

            # Determine if we were "right"
            if direction == "BUY_YES":
                we_predicted_yes = True
                correct = (outcome == "YES")
            else:
                we_predicted_yes = False
                correct = (outcome == "NO")

            # Determine bucket
            if prob < 0.2:
                bucket = "0-20%"
            elif prob < 0.4:
                bucket = "20-40%"
            elif prob < 0.6:
                bucket = "40-60%"
            elif prob < 0.8:
                bucket = "60-80%"
            else:
                bucket = "80-100%"

            buckets[bucket]["predictions"] += 1
            if correct:
                buckets[bucket]["correct"] += 1

        conn.close()

        # Calculate accuracy for each bucket
        accuracy = {}
        for bucket, data in buckets.items():
            if data["predictions"] > 0:
                accuracy[bucket] = data["correct"] / data["predictions"]
            else:
                accuracy[bucket] = None

        return accuracy

    def generate_summary_text(self) -> str:
        """Generate a text summary of performance."""
        metrics = self.get_overall_metrics()

        lines = [
            "=" * 60,
            "PERFORMANCE SUMMARY",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 60,
            "",
            "OVERALL RESULTS",
            "-" * 40,
            f"Total Trades:     {metrics.total_trades}",
            f"Wins:             {metrics.wins}",
            f"Losses:           {metrics.losses}",
            f"Win Rate:         {format_percent(metrics.win_rate)}",
            "",
            f"Gross P&L:        {format_currency(metrics.gross_pnl)}",
            f"Total Fees:       {format_currency(metrics.total_fees)}",
            f"Net P&L:          {format_currency(metrics.net_pnl)}",
            f"Avg P&L/Trade:    {format_currency(metrics.avg_pnl_per_trade)}",
            "",
            f"Largest Win:      {format_currency(metrics.largest_win)}",
            f"Largest Loss:     {format_currency(metrics.largest_loss)}",
            f"Max Drawdown:     {format_currency(metrics.max_drawdown)}",
            "",
            f"Avg Edge:         {format_percent(metrics.avg_edge_at_entry)}",
        ]

        if metrics.sharpe_ratio is not None:
            lines.append(f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
        if metrics.profit_factor is not None:
            lines.append(f"Profit Factor:    {metrics.profit_factor:.2f}")

        # Add breakdown by confidence
        by_conf = self.get_performance_by_confidence()
        if by_conf:
            lines.extend([
                "",
                "BY CONFIDENCE LEVEL",
                "-" * 40
            ])
            for conf, data in sorted(by_conf.items()):
                lines.append(
                    f"  {conf:8}: {data['trades']:3} trades, "
                    f"{format_percent(data['win_rate']):6} win rate, "
                    f"{format_currency(data['net_pnl']):8} P&L"
                )

        # Add breakdown by edge bucket
        by_edge = self.get_performance_by_edge_bucket()
        if by_edge:
            lines.extend([
                "",
                "BY EDGE BUCKET",
                "-" * 40
            ])
            for bucket, data in by_edge.items():
                lines.append(
                    f"  {bucket:8}: {data['trades']:3} trades, "
                    f"{format_percent(data['win_rate']):6} win rate, "
                    f"{format_currency(data['net_pnl']):8} P&L"
                )

        lines.append("=" * 60)

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    print("Testing PerformanceAnalyzer...")

    analyzer = PerformanceAnalyzer()

    # Get metrics
    metrics = analyzer.get_overall_metrics()
    print(f"\nTotal trades: {metrics.total_trades}")
    print(f"Win rate: {format_percent(metrics.win_rate)}")
    print(f"Net P&L: {format_currency(metrics.net_pnl)}")

    # Generate report
    print("\n" + analyzer.generate_summary_text())
