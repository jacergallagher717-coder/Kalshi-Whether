"""
Backtest Framework - Validate strategy against historical data.

This module:
1. Loads historical trades and their outcomes
2. Simulates strategy with different parameters
3. Calculates performance metrics (win rate, P&L, Sharpe ratio)
4. Helps optimize edge thresholds and position sizing

Usage:
    backtest = Backtester()
    results = backtest.run_backtest(start_date, end_date)
    print(backtest.generate_report(results))
"""

import sqlite3
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import statistics
import os

from config.settings import (
    DATABASE_PATH, MIN_EDGE_THRESHOLD, HIGH_EDGE_THRESHOLD,
    HIGH_EDGE_POSITION_PERCENT, NORMAL_POSITION_PERCENT,
    MAX_FORECAST_DAYS, NEXT_DAY_EDGE_PENALTY
)
from utils.logger import get_logger

logger = get_logger("backtest")

TRADES_DB = os.path.join(DATABASE_PATH, "trades.db")


@dataclass
class BacktestTrade:
    """A single trade in the backtest."""
    ticker: str
    city: str
    target_date: date
    direction: str
    entry_price: float
    contracts: int
    edge_at_entry: float
    our_probability: float
    market_probability: float
    outcome: Optional[str] = None  # "WIN" or "LOSS"
    actual_temp: Optional[float] = None
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    start_date: date
    end_date: date
    total_trades: int
    wins: int
    losses: int
    pending: int
    total_pnl: float
    win_rate: float
    avg_edge: float
    avg_pnl_per_trade: float
    sharpe_ratio: float
    max_drawdown: float
    trades: List[BacktestTrade] = field(default_factory=list)
    pnl_by_city: Dict[str, float] = field(default_factory=dict)
    pnl_by_edge_bucket: Dict[str, float] = field(default_factory=dict)


class Backtester:
    """
    Backtest trading strategy against historical data.

    Uses actual trades from the database to calculate performance metrics.
    Can also simulate "what-if" scenarios with different parameters.
    """

    def __init__(self, db_path: str = None):
        """Initialize the backtester."""
        self.db_path = db_path or TRADES_DB
        logger.info("Backtester initialized")

    def load_historical_trades(
        self,
        start_date: date = None,
        end_date: date = None,
        status: str = None
    ) -> List[BacktestTrade]:
        """
        Load historical trades from the database.

        Args:
            start_date: Start of date range
            end_date: End of date range
            status: Filter by status ('SETTLED', 'OPEN', etc.)

        Returns:
            List of BacktestTrade objects
        """
        if not os.path.exists(self.db_path):
            logger.warning(f"Trades database not found: {self.db_path}")
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT
                ticker, direction, entry_price, contracts,
                edge_at_entry, status, settlement_result,
                actual_temp, net_pnl, created_at, target_date
            FROM trades
            WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND DATE(created_at) >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND DATE(created_at) <= ?"
            params.append(end_date.isoformat())
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not query trades: {e}")
            conn.close()
            return []

        conn.close()

        trades = []
        for row in rows:
            ticker, direction, entry_price, contracts, edge, status, result, actual, pnl, created, target = row

            # Extract city from ticker (e.g., KXHIGHNYC-26JAN07-T45 -> NYC)
            city = "UNK"
            if ticker:
                parts = ticker.split('-')
                if parts:
                    city_part = parts[0].replace('KXHIGH', '').replace('KXLOW', '')
                    city = city_part if city_part else "UNK"

            # Parse target date
            target_dt = None
            if target:
                try:
                    target_dt = datetime.fromisoformat(target).date() if isinstance(target, str) else target
                except:
                    pass

            trades.append(BacktestTrade(
                ticker=ticker or "",
                city=city,
                target_date=target_dt or date.today(),
                direction=direction or "",
                entry_price=entry_price or 0,
                contracts=contracts or 0,
                edge_at_entry=edge or 0,
                our_probability=(entry_price or 0) + (edge or 0),  # Approximate
                market_probability=entry_price or 0,
                outcome="WIN" if result == "correct" else ("LOSS" if result else None),
                actual_temp=actual,
                pnl=pnl or 0
            ))

        return trades

    def run_backtest(
        self,
        start_date: date = None,
        end_date: date = None,
        min_edge: float = None,
        simulated_bankroll: float = 10000
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            min_edge: Minimum edge threshold (uses current settings if None)
            simulated_bankroll: Starting bankroll for position sizing

        Returns:
            BacktestResult with all metrics
        """
        start_date = start_date or (date.today() - timedelta(days=30))
        end_date = end_date or date.today()
        min_edge = min_edge if min_edge is not None else MIN_EDGE_THRESHOLD

        logger.info(f"Running backtest from {start_date} to {end_date} with min_edge={min_edge:.0%}")

        # Load all settled trades
        all_trades = self.load_historical_trades(start_date, end_date)

        # Filter by edge threshold
        filtered_trades = [t for t in all_trades if t.edge_at_entry >= min_edge]

        # Separate by outcome
        settled = [t for t in filtered_trades if t.outcome]
        wins = [t for t in settled if t.outcome == "WIN"]
        losses = [t for t in settled if t.outcome == "LOSS"]
        pending = [t for t in filtered_trades if not t.outcome]

        # Calculate metrics
        total_pnl = sum(t.pnl for t in settled)
        win_rate = len(wins) / len(settled) if settled else 0
        avg_edge = statistics.mean([t.edge_at_entry for t in filtered_trades]) if filtered_trades else 0
        avg_pnl = total_pnl / len(settled) if settled else 0

        # Calculate Sharpe ratio (if we have enough data)
        sharpe = 0.0
        if len(settled) >= 5:
            pnls = [t.pnl for t in settled]
            if statistics.stdev(pnls) > 0:
                sharpe = statistics.mean(pnls) / statistics.stdev(pnls) * (252 ** 0.5)  # Annualized

        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown(settled)

        # P&L by city
        pnl_by_city = {}
        for t in settled:
            if t.city not in pnl_by_city:
                pnl_by_city[t.city] = 0
            pnl_by_city[t.city] += t.pnl

        # P&L by edge bucket
        pnl_by_edge = {"30-40%": 0, "40-50%": 0, "50%+": 0}
        for t in settled:
            edge_pct = t.edge_at_entry * 100
            if edge_pct >= 50:
                pnl_by_edge["50%+"] += t.pnl
            elif edge_pct >= 40:
                pnl_by_edge["40-50%"] += t.pnl
            else:
                pnl_by_edge["30-40%"] += t.pnl

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_trades=len(filtered_trades),
            wins=len(wins),
            losses=len(losses),
            pending=len(pending),
            total_pnl=total_pnl,
            win_rate=win_rate,
            avg_edge=avg_edge,
            avg_pnl_per_trade=avg_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            trades=filtered_trades,
            pnl_by_city=pnl_by_city,
            pnl_by_edge_bucket=pnl_by_edge
        )

    def _calculate_max_drawdown(self, trades: List[BacktestTrade]) -> float:
        """Calculate maximum drawdown from trade sequence."""
        if not trades:
            return 0.0

        cumulative = 0
        peak = 0
        max_dd = 0

        for t in trades:
            cumulative += t.pnl
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_dd = max(max_dd, drawdown)

        return max_dd

    def optimize_parameters(
        self,
        start_date: date = None,
        end_date: date = None
    ) -> Dict[str, any]:
        """
        Find optimal parameters by backtesting different settings.

        Args:
            start_date: Start date for optimization
            end_date: End date for optimization

        Returns:
            Dict with optimal parameters and their metrics
        """
        start_date = start_date or (date.today() - timedelta(days=30))
        end_date = end_date or date.today()

        edge_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        best_result = None
        best_edge = None
        best_sharpe = float('-inf')

        results = []

        for edge in edge_thresholds:
            result = self.run_backtest(start_date, end_date, min_edge=edge)

            # Score by Sharpe ratio (balances returns and risk)
            score = result.sharpe_ratio if result.total_trades >= 5 else float('-inf')

            results.append({
                "edge_threshold": edge,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "sharpe_ratio": result.sharpe_ratio
            })

            if score > best_sharpe and result.total_trades >= 5:
                best_sharpe = score
                best_result = result
                best_edge = edge

        return {
            "optimal_edge_threshold": best_edge,
            "optimal_result": best_result,
            "all_results": results
        }

    def generate_report(self, result: BacktestResult = None) -> str:
        """
        Generate a text report of backtest results.

        Args:
            result: BacktestResult to report on (runs default backtest if None)

        Returns:
            Formatted text report
        """
        if result is None:
            result = self.run_backtest()

        lines = []
        lines.append("=" * 60)
        lines.append("BACKTEST REPORT")
        lines.append("=" * 60)
        lines.append(f"Period: {result.start_date} to {result.end_date}")
        lines.append("")

        lines.append("PERFORMANCE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Trades:     {result.total_trades}")
        lines.append(f"Wins:             {result.wins}")
        lines.append(f"Losses:           {result.losses}")
        lines.append(f"Pending:          {result.pending}")
        lines.append(f"Win Rate:         {result.win_rate:.1%}")
        lines.append("")
        lines.append(f"Total P&L:        ${result.total_pnl:+.2f}")
        lines.append(f"Avg P&L/Trade:    ${result.avg_pnl_per_trade:+.2f}")
        lines.append(f"Avg Edge:         {result.avg_edge:.1%}")
        lines.append(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        lines.append(f"Max Drawdown:     ${result.max_drawdown:.2f}")

        if result.pnl_by_city:
            lines.append("")
            lines.append("P&L BY CITY")
            lines.append("-" * 40)
            for city, pnl in sorted(result.pnl_by_city.items(), key=lambda x: -x[1]):
                lines.append(f"  {city}: ${pnl:+.2f}")

        if result.pnl_by_edge_bucket:
            lines.append("")
            lines.append("P&L BY EDGE BUCKET")
            lines.append("-" * 40)
            for bucket, pnl in result.pnl_by_edge_bucket.items():
                lines.append(f"  {bucket}: ${pnl:+.2f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def compare_strategies(
        self,
        start_date: date = None,
        end_date: date = None
    ) -> str:
        """
        Compare different strategy configurations.

        Returns:
            Report comparing strategies
        """
        start_date = start_date or (date.today() - timedelta(days=30))
        end_date = end_date or date.today()

        strategies = {
            "Conservative (40% edge)": 0.40,
            "Balanced (30% edge)": 0.30,
            "Aggressive (20% edge)": 0.20
        }

        lines = []
        lines.append("=" * 70)
        lines.append("STRATEGY COMPARISON")
        lines.append("=" * 70)
        lines.append(f"Period: {start_date} to {end_date}")
        lines.append("")
        lines.append(f"{'Strategy':<25} {'Trades':<8} {'Win%':<8} {'P&L':<12} {'Sharpe':<8}")
        lines.append("-" * 70)

        for name, edge in strategies.items():
            result = self.run_backtest(start_date, end_date, min_edge=edge)
            lines.append(
                f"{name:<25} {result.total_trades:<8} {result.win_rate*100:<7.0f}% "
                f"${result.total_pnl:<10.2f} {result.sharpe_ratio:<8.2f}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Weather Trading Strategy")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest")
    parser.add_argument("--optimize", action="store_true", help="Optimize parameters")
    parser.add_argument("--compare", action="store_true", help="Compare strategies")
    args = parser.parse_args()

    backtest = Backtester()

    start = date.today() - timedelta(days=args.days)
    end = date.today()

    if args.optimize:
        print("\nOptimizing parameters...")
        opt = backtest.optimize_parameters(start, end)

        print(f"\nOptimal Edge Threshold: {opt['optimal_edge_threshold']:.0%}")
        print("\nAll tested thresholds:")
        for r in opt['all_results']:
            print(f"  {r['edge_threshold']:.0%}: {r['total_trades']} trades, "
                  f"{r['win_rate']:.0%} win rate, ${r['total_pnl']:.2f} P&L")

    elif args.compare:
        print(backtest.compare_strategies(start, end))

    else:
        result = backtest.run_backtest(start, end)
        print(backtest.generate_report(result))
