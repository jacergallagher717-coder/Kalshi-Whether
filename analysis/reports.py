"""
Report generation for daily and weekly trading summaries.

Generates:
- Daily performance reports
- Weekly summary reports
- Model accuracy analysis
- Trade-by-trade logs
"""

import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import json

from config.settings import DATABASE_PATH
from analysis.performance import PerformanceAnalyzer
from analysis.edge_validation import EdgeValidator
from trading.paper_trader import PaperTrader
from trading.position_manager import PositionManager
from utils.logger import get_logger
from utils.helpers import format_currency, format_percent

logger = get_logger("reports")

REPORTS_DIR = os.path.join(DATABASE_PATH, "reports")


class ReportGenerator:
    """
    Generates trading reports for analysis and record-keeping.

    Report types:
    - Daily: End-of-day summary
    - Weekly: Week-in-review with validation analysis
    - Trade log: Detailed trade-by-trade export
    """

    def __init__(self):
        """Initialize the report generator."""
        self.analyzer = PerformanceAnalyzer()
        self.validator = EdgeValidator()
        self.trader = PaperTrader()
        self.position_manager = PositionManager()

        # Ensure reports directory exists
        os.makedirs(REPORTS_DIR, exist_ok=True)

        logger.info("Report generator initialized")

    def generate_daily_report(self, report_date: date = None) -> Dict:
        """
        Generate daily performance report.

        Args:
            report_date: Date to report on (defaults to today)

        Returns:
            Dictionary with daily statistics
        """
        report_date = report_date or date.today()

        # Get trades from today
        trades = self.trader.get_all_trades()
        today_trades = [
            t for t in trades
            if t.created_at.date() == report_date or
            (t.settled_at and t.settled_at.date() == report_date)
        ]

        # Separate by status
        executed_today = [t for t in today_trades if t.created_at.date() == report_date]
        settled_today = [t for t in today_trades if t.settled_at and t.settled_at.date() == report_date]

        # Calculate today's P&L
        today_gross_pnl = sum(t.gross_pnl or 0 for t in settled_today)
        today_fees = sum(t.simulated_fees for t in settled_today)
        today_net_pnl = sum(t.net_pnl or 0 for t in settled_today)
        today_wins = sum(1 for t in settled_today if t.net_pnl and t.net_pnl > 0)

        # Get open positions
        open_positions = self.position_manager.get_open_positions()

        report = {
            "date": report_date.isoformat(),
            "generated_at": datetime.utcnow().isoformat(),
            "trades_executed": len(executed_today),
            "trades_settled": len(settled_today),
            "gross_pnl": today_gross_pnl,
            "fees_paid": today_fees,
            "net_pnl": today_net_pnl,
            "win_rate": today_wins / len(settled_today) if settled_today else 0,
            "avg_edge": (
                sum(t.edge_at_entry for t in executed_today) / len(executed_today)
                if executed_today else 0
            ),
            "open_positions": len(open_positions),
            "total_exposure": sum(p.entry_price * p.contracts for p in open_positions),
            "executed_trades": [
                {
                    "id": t.id,
                    "ticker": t.ticker,
                    "direction": t.direction,
                    "contracts": t.contracts,
                    "entry_price": t.entry_price,
                    "edge": t.edge_at_entry,
                    "confidence": t.confidence
                }
                for t in executed_today
            ],
            "settled_trades": [
                {
                    "id": t.id,
                    "ticker": t.ticker,
                    "direction": t.direction,
                    "outcome": t.settlement_outcome,
                    "actual_temp": t.actual_temp,
                    "net_pnl": t.net_pnl
                }
                for t in settled_today
            ]
        }

        # Save report
        self._save_report(report, f"daily_{report_date.isoformat()}.json")

        logger.info(f"Generated daily report for {report_date}: {len(settled_today)} settled, {format_currency(today_net_pnl)} P&L")

        return report

    def generate_weekly_report(self, week_ending: date = None) -> Dict:
        """
        Generate weekly summary report with validation analysis.

        Args:
            week_ending: Last day of the week (defaults to last Sunday)

        Returns:
            Dictionary with weekly statistics and validation
        """
        if week_ending is None:
            # Find last Sunday
            today = date.today()
            days_since_sunday = today.weekday() + 1
            if days_since_sunday == 7:
                days_since_sunday = 0
            week_ending = today - timedelta(days=days_since_sunday)

        week_start = week_ending - timedelta(days=6)

        # Get trades from this week
        all_trades = self.trader.get_all_trades()
        week_trades = [
            t for t in all_trades
            if (t.created_at.date() >= week_start and t.created_at.date() <= week_ending) or
            (t.settled_at and t.settled_at.date() >= week_start and t.settled_at.date() <= week_ending)
        ]

        settled_this_week = [
            t for t in week_trades
            if t.settled_at and t.settled_at.date() >= week_start
        ]

        # Calculate weekly stats
        week_net_pnl = sum(t.net_pnl or 0 for t in settled_this_week)
        week_wins = sum(1 for t in settled_this_week if t.net_pnl and t.net_pnl > 0)

        # Get overall metrics and validation
        overall_metrics = self.analyzer.get_overall_metrics()
        validation = self.validator.get_validation_summary()

        # Performance by confidence
        by_confidence = self.analyzer.get_performance_by_confidence()

        # Performance by edge bucket
        by_edge = self.analyzer.get_performance_by_edge_bucket()

        report = {
            "week_start": week_start.isoformat(),
            "week_end": week_ending.isoformat(),
            "generated_at": datetime.utcnow().isoformat(),

            # This week
            "this_week": {
                "trades_executed": len([t for t in week_trades if t.created_at.date() >= week_start]),
                "trades_settled": len(settled_this_week),
                "net_pnl": week_net_pnl,
                "wins": week_wins,
                "win_rate": week_wins / len(settled_this_week) if settled_this_week else 0
            },

            # All time
            "all_time": {
                "total_trades": overall_metrics.total_trades,
                "win_rate": overall_metrics.win_rate,
                "net_pnl": overall_metrics.net_pnl,
                "avg_edge": overall_metrics.avg_edge_at_entry,
                "sharpe_ratio": overall_metrics.sharpe_ratio,
                "max_drawdown": overall_metrics.max_drawdown
            },

            # Validation status
            "validation": {
                "status": validation["status"],
                "recommendation": validation["recommendation"],
                "tests_passed": validation["tests_passed"],
                "tests_total": validation["tests_total"],
                "sample_size": validation["sample_size"],
                "required_sample_size": validation["required_sample_size"]
            },

            # Breakdowns
            "by_confidence": by_confidence,
            "by_edge_bucket": by_edge
        }

        # Save report
        self._save_report(report, f"weekly_{week_ending.isoformat()}.json")

        logger.info(
            f"Generated weekly report for {week_start} to {week_ending}: "
            f"{len(settled_this_week)} trades, {format_currency(week_net_pnl)} P&L, "
            f"Validation: {validation['status']}"
        )

        return report

    def generate_trade_log(self, start_date: date = None, end_date: date = None) -> List[Dict]:
        """
        Generate detailed trade log export.

        Args:
            start_date: Start of date range (defaults to all time)
            end_date: End of date range (defaults to today)

        Returns:
            List of trade dictionaries
        """
        trades = self.trader.get_all_trades()

        # Filter by date
        if start_date:
            trades = [t for t in trades if t.created_at.date() >= start_date]
        if end_date:
            trades = [t for t in trades if t.created_at.date() <= end_date]

        log = []
        for t in trades:
            log.append({
                "id": t.id,
                "created_at": t.created_at.isoformat(),
                "ticker": t.ticker,
                "location": t.location,
                "target_date": t.target_date.isoformat(),
                "temp_threshold": t.temp_threshold,
                "direction": t.direction,
                "contracts": t.contracts,
                "entry_price": t.entry_price,
                "simulated_fees": t.simulated_fees,
                "our_probability": t.our_probability,
                "edge_at_entry": t.edge_at_entry,
                "confidence": t.confidence,
                "reasoning": t.reasoning,
                "status": t.status,
                "settlement_outcome": t.settlement_outcome,
                "actual_temp": t.actual_temp,
                "exit_price": t.exit_price,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "settled_at": t.settled_at.isoformat() if t.settled_at else None
            })

        # Save export
        filename = "trade_log"
        if start_date:
            filename += f"_from_{start_date.isoformat()}"
        if end_date:
            filename += f"_to_{end_date.isoformat()}"
        filename += ".json"

        self._save_report(log, filename)

        logger.info(f"Generated trade log with {len(log)} trades")

        return log

    def generate_daily_text_report(self, report_date: date = None) -> str:
        """Generate human-readable daily report."""
        report = self.generate_daily_report(report_date)

        lines = [
            "=" * 60,
            f"DAILY REPORT: {report['date']}",
            f"Generated: {report['generated_at']}",
            "=" * 60,
            "",
            "TODAY'S ACTIVITY",
            "-" * 40,
            f"Trades Executed:  {report['trades_executed']}",
            f"Trades Settled:   {report['trades_settled']}",
            f"Gross P&L:        {format_currency(report['gross_pnl'])}",
            f"Fees Paid:        {format_currency(report['fees_paid'])}",
            f"Net P&L:          {format_currency(report['net_pnl'])}",
            f"Win Rate:         {format_percent(report['win_rate'])}",
            f"Avg Edge:         {format_percent(report['avg_edge'])}",
            "",
            "CURRENT EXPOSURE",
            "-" * 40,
            f"Open Positions:   {report['open_positions']}",
            f"Total Exposure:   {format_currency(report['total_exposure'])}",
        ]

        if report['executed_trades']:
            lines.extend([
                "",
                "TRADES EXECUTED TODAY",
                "-" * 40
            ])
            for t in report['executed_trades']:
                lines.append(
                    f"  {t['ticker']}: {t['direction']} x{t['contracts']} @ "
                    f"{format_currency(t['entry_price'])} | Edge: {format_percent(t['edge'])}"
                )

        if report['settled_trades']:
            lines.extend([
                "",
                "TRADES SETTLED TODAY",
                "-" * 40
            ])
            for t in report['settled_trades']:
                lines.append(
                    f"  {t['ticker']}: {t['outcome']} @ {t['actual_temp']}°F | "
                    f"P&L: {format_currency(t['net_pnl'])}"
                )

        lines.append("=" * 60)

        return "\n".join(lines)

    def generate_weekly_text_report(self, week_ending: date = None) -> str:
        """Generate human-readable weekly report."""
        report = self.generate_weekly_report(week_ending)

        lines = [
            "=" * 60,
            f"WEEKLY REPORT: {report['week_start']} to {report['week_end']}",
            f"Generated: {report['generated_at']}",
            "=" * 60,
            "",
            "THIS WEEK",
            "-" * 40,
            f"Trades Executed:  {report['this_week']['trades_executed']}",
            f"Trades Settled:   {report['this_week']['trades_settled']}",
            f"Net P&L:          {format_currency(report['this_week']['net_pnl'])}",
            f"Win Rate:         {format_percent(report['this_week']['win_rate'])}",
            "",
            "ALL TIME",
            "-" * 40,
            f"Total Trades:     {report['all_time']['total_trades']}",
            f"Win Rate:         {format_percent(report['all_time']['win_rate'])}",
            f"Net P&L:          {format_currency(report['all_time']['net_pnl'])}",
            f"Avg Edge:         {format_percent(report['all_time']['avg_edge'])}",
            f"Max Drawdown:     {format_currency(report['all_time']['max_drawdown'])}",
        ]

        if report['all_time']['sharpe_ratio']:
            lines.append(f"Sharpe Ratio:     {report['all_time']['sharpe_ratio']:.2f}")

        lines.extend([
            "",
            "EDGE VALIDATION",
            "-" * 40,
            f"Status:           {report['validation']['status']}",
            f"Tests Passed:     {report['validation']['tests_passed']}/{report['validation']['tests_total']}",
            f"Sample Size:      {report['validation']['sample_size']} trades",
            f"Required:         ~{report['validation']['required_sample_size']} trades",
            "",
            f"Recommendation: {report['validation']['recommendation']}",
        ])

        if report['by_confidence']:
            lines.extend([
                "",
                "BY CONFIDENCE LEVEL",
                "-" * 40
            ])
            for conf, data in sorted(report['by_confidence'].items()):
                lines.append(
                    f"  {conf:8}: {data['trades']:3} trades, "
                    f"{format_percent(data['win_rate'])} win, "
                    f"{format_currency(data['net_pnl'])} P&L"
                )

        lines.append("=" * 60)

        return "\n".join(lines)

    def _save_report(self, data: any, filename: str):
        """Save report to file."""
        filepath = os.path.join(REPORTS_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"Saved report to {filepath}")


# Example usage
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    print("Testing ReportGenerator...")

    generator = ReportGenerator()

    # Generate daily report
    print("\n" + generator.generate_daily_text_report())

    # Generate weekly report
    print("\n" + generator.generate_weekly_text_report())
