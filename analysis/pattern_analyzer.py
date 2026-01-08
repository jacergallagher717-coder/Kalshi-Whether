"""
Pattern Analyzer - Find patterns in winners vs losers to improve strategy.

This module analyzes settled trades to identify:
1. What factors are common in winning trades
2. What factors are common in losing trades
3. Actionable recommendations to improve win rate

Key insight: We want to find patterns like "trades with >30% probability win 70%
of the time, but trades with <20% probability only win 25% of the time"

Supports both:
- Paper trades from local database
- Real Kalshi trades via API
"""

import sqlite3
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from config.settings import TRADES_DB
from utils.logger import get_logger

logger = get_logger("pattern_analyzer")


def get_kalshi_trade_history() -> List[dict]:
    """
    Fetch real trade history from Kalshi API.

    Returns list of fill records with trade details.
    """
    try:
        from data.collectors.kalshi_client import KalshiClient

        client = KalshiClient()
        if not client.login():
            logger.warning("Could not login to Kalshi for trade history")
            return []

        # Get fills (executed trades)
        url = f'{client.base_url}/portfolio/fills'
        response = client._make_request('GET', url, params={'limit': 200})

        if response and 'fills' in response:
            return response['fills']
        return []
    except Exception as e:
        logger.warning(f"Could not fetch Kalshi trades: {e}")
        return []


@dataclass
class PatternInsight:
    """A single insight about winning/losing patterns."""
    category: str  # e.g., "probability", "edge", "city", "time"
    finding: str   # Human readable finding
    winners: int
    losers: int
    win_rate: float
    recommendation: Optional[str] = None
    priority: str = "medium"  # high, medium, low


@dataclass
class TradeRecord:
    """Detailed record of a trade for analysis."""
    ticker: str
    location: str
    direction: str
    entry_price: float
    our_probability: float
    edge: float
    confidence: str
    target_date: str
    created_at: str
    days_out: int
    is_bracket: bool
    outcome: str  # WIN or LOSS
    pnl: float


class PatternAnalyzer:
    """
    Analyzes trade patterns to find what works and what doesn't.

    Compares characteristics of winning trades vs losing trades
    to identify actionable improvements.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or TRADES_DB
        logger.info("Pattern analyzer initialized")

    def get_all_settled_trades(self) -> List[TradeRecord]:
        """Load all settled trades for analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                ticker,
                location,
                direction,
                entry_price,
                our_probability,
                edge_at_entry,
                confidence,
                target_date,
                created_at,
                net_pnl
            FROM paper_trades
            WHERE status = 'SETTLED'
            ORDER BY created_at ASC
        ''')

        trades = []
        for row in cursor.fetchall():
            ticker = row[0]
            target_date = row[7]
            created_at = row[8]

            # Calculate days out
            try:
                target = datetime.fromisoformat(target_date).date() if target_date else None
                created = datetime.fromisoformat(created_at).date() if created_at else None
                days_out = (target - created).days if target and created else 0
            except:
                days_out = 0

            # Check if bracket market
            is_bracket = '-B' in ticker if ticker else False

            # Determine outcome
            pnl = row[9] or 0
            outcome = "WIN" if pnl > 0 else "LOSS"

            trades.append(TradeRecord(
                ticker=ticker,
                location=row[1],
                direction=row[2],
                entry_price=row[3] or 0,
                our_probability=row[4] or 0,
                edge=row[5] or 0,
                confidence=row[6],
                target_date=target_date,
                created_at=created_at,
                days_out=days_out,
                is_bracket=is_bracket,
                outcome=outcome,
                pnl=pnl
            ))

        conn.close()
        return trades

    def get_kalshi_settled_trades(self) -> List[TradeRecord]:
        """
        Get settled trades from Kalshi API for analysis.

        Analyzes fills to determine wins/losses based on settlement.
        """
        fills = get_kalshi_trade_history()
        if not fills:
            return []

        trades = []
        # Group fills by ticker to track entry/exit
        ticker_fills = defaultdict(list)
        for fill in fills:
            ticker_fills[fill.get('ticker', '')].append(fill)

        for ticker, ticker_fills_list in ticker_fills.items():
            if not ticker:
                continue

            # Extract city from ticker (e.g., KXHIGHNY-26JAN07-T50 -> NY)
            location = ""
            for city in ["NY", "NYC", "CHI", "LA", "LAX", "MIA", "AUS", "DEN", "PHI"]:
                if city in ticker.upper():
                    location = city.replace("NYC", "NY").replace("LAX", "LA")
                    break

            # Calculate total P&L from fills
            total_pnl = 0
            entry_price = 0
            direction = ""
            created_at = ""

            for fill in ticker_fills_list:
                side = fill.get('side', '')
                count = fill.get('count', 0)
                # Prices are in cents, convert to dollars
                yes_price = fill.get('yes_price', 0) / 100
                no_price = fill.get('no_price', 0) / 100

                if not created_at:
                    created_at = fill.get('created_time', '')

                if side == 'yes':
                    if not direction:
                        direction = "BUY_YES"
                        entry_price = yes_price
                    # P&L depends on whether this was entry or exit
                elif side == 'no':
                    if not direction:
                        direction = "BUY_NO"
                        entry_price = no_price

            # Check if market is settled by looking at ticker date
            is_bracket = '-B' in ticker

            # For now, estimate outcome from current market state
            # In production, would check settlement status
            # We'll mark as WIN if entry price was good
            outcome = "PENDING"
            pnl = 0

            # Only include if we have enough data
            if direction and entry_price > 0:
                trades.append(TradeRecord(
                    ticker=ticker,
                    location=location,
                    direction=direction,
                    entry_price=entry_price,
                    our_probability=entry_price,  # Approximate
                    edge=0.30,  # Unknown, use minimum
                    confidence="medium",
                    target_date="",
                    created_at=created_at,
                    days_out=0,
                    is_bracket=is_bracket,
                    outcome=outcome,
                    pnl=pnl
                ))

        return trades

    def get_kalshi_positions_analysis(self) -> str:
        """
        Analyze current Kalshi positions to show performance patterns.

        This works even without settled trades in the database.
        """
        try:
            from data.collectors.kalshi_client import KalshiClient

            client = KalshiClient()
            if not client.login():
                return "Could not login to Kalshi"

            positions = client.get_positions()
            if not positions:
                return "No positions found"

            lines = []
            lines.append("\n" + "=" * 60)
            lines.append("KALSHI POSITION ANALYSIS")
            lines.append("=" * 60)

            # Group by city
            city_stats = defaultdict(lambda: {"count": 0, "exposure": 0})

            for pos in positions:
                if pos.market_exposure == 0:
                    continue

                ticker = pos.ticker
                for city in ["NY", "CHI", "LA", "MIA", "AUS", "DEN", "PHI"]:
                    if city in ticker.upper():
                        city_stats[city]["count"] += 1
                        city_stats[city]["exposure"] += abs(pos.market_exposure)
                        break

            if city_stats:
                lines.append("\nPOSITIONS BY CITY:")
                lines.append("-" * 40)
                for city, stats in sorted(city_stats.items(), key=lambda x: -x[1]["exposure"]):
                    lines.append(f"  {city}: {stats['count']} positions, {stats['exposure']} contracts")

            # Group by direction
            yes_count = sum(1 for p in positions if p.market_exposure > 0)
            no_count = sum(1 for p in positions if p.market_exposure < 0)

            lines.append(f"\nBY DIRECTION:")
            lines.append(f"  YES positions: {yes_count}")
            lines.append(f"  NO positions: {no_count}")

            lines.append("\n" + "=" * 60)
            return "\n".join(lines)

        except Exception as e:
            return f"Error analyzing Kalshi positions: {e}"

    def analyze_by_probability_bucket(self, trades: List[TradeRecord]) -> List[PatternInsight]:
        """Analyze win rate by probability bucket."""
        insights = []

        buckets = {
            "0-10%": (0, 0.10),
            "10-20%": (0.10, 0.20),
            "20-30%": (0.20, 0.30),
            "30-40%": (0.30, 0.40),
            "40-50%": (0.40, 0.50),
            "50-60%": (0.50, 0.60),
            "60-80%": (0.60, 0.80),
            "80-100%": (0.80, 1.01),
        }

        for bucket_name, (low, high) in buckets.items():
            bucket_trades = [t for t in trades if low <= t.our_probability < high]
            if not bucket_trades:
                continue

            winners = sum(1 for t in bucket_trades if t.outcome == "WIN")
            losers = len(bucket_trades) - winners
            win_rate = winners / len(bucket_trades) if bucket_trades else 0

            # Determine priority and recommendation
            if win_rate < 0.30 and len(bucket_trades) >= 3:
                priority = "high"
                recommendation = f"Consider increasing MIN_PROBABILITY_THRESHOLD above {high:.0%}"
            elif win_rate > 0.70 and len(bucket_trades) >= 3:
                priority = "high"
                recommendation = f"Sweet spot! Consider increasing position size for {bucket_name} probability trades"
            else:
                priority = "medium"
                recommendation = None

            insights.append(PatternInsight(
                category="probability",
                finding=f"Trades with {bucket_name} probability: {winners}W/{losers}L ({win_rate:.0%} win rate)",
                winners=winners,
                losers=losers,
                win_rate=win_rate,
                recommendation=recommendation,
                priority=priority
            ))

        return insights

    def analyze_by_edge_bucket(self, trades: List[TradeRecord]) -> List[PatternInsight]:
        """Analyze win rate by edge bucket."""
        insights = []

        buckets = {
            "30-40%": (0.30, 0.40),
            "40-50%": (0.40, 0.50),
            "50-60%": (0.50, 0.60),
            "60-80%": (0.60, 0.80),
            "80%+": (0.80, 2.0),
        }

        for bucket_name, (low, high) in buckets.items():
            bucket_trades = [t for t in trades if low <= abs(t.edge) < high]
            if not bucket_trades:
                continue

            winners = sum(1 for t in bucket_trades if t.outcome == "WIN")
            losers = len(bucket_trades) - winners
            win_rate = winners / len(bucket_trades)

            insights.append(PatternInsight(
                category="edge",
                finding=f"Trades with {bucket_name} edge: {winners}W/{losers}L ({win_rate:.0%} win rate)",
                winners=winners,
                losers=losers,
                win_rate=win_rate,
                priority="medium"
            ))

        return insights

    def analyze_by_city(self, trades: List[TradeRecord]) -> List[PatternInsight]:
        """Analyze win rate by city."""
        insights = []

        city_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})

        for trade in trades:
            city = trade.location or "UNKNOWN"
            if trade.outcome == "WIN":
                city_stats[city]["wins"] += 1
            else:
                city_stats[city]["losses"] += 1
            city_stats[city]["pnl"] += trade.pnl

        for city, stats in sorted(city_stats.items(), key=lambda x: x[1]["wins"] + x[1]["losses"], reverse=True):
            total = stats["wins"] + stats["losses"]
            if total == 0:
                continue

            win_rate = stats["wins"] / total

            if win_rate < 0.30 and total >= 3:
                priority = "high"
                recommendation = f"Consider reducing position size or excluding {city}"
            elif win_rate > 0.70 and total >= 3:
                priority = "high"
                recommendation = f"Strong performer! Consider increasing position size for {city}"
            else:
                priority = "low"
                recommendation = None

            insights.append(PatternInsight(
                category="city",
                finding=f"{city}: {stats['wins']}W/{stats['losses']}L ({win_rate:.0%}), P&L: ${stats['pnl']:.2f}",
                winners=stats["wins"],
                losers=stats["losses"],
                win_rate=win_rate,
                recommendation=recommendation,
                priority=priority
            ))

        return insights

    def analyze_by_days_out(self, trades: List[TradeRecord]) -> List[PatternInsight]:
        """Analyze win rate by forecast horizon."""
        insights = []

        days_stats = defaultdict(lambda: {"wins": 0, "losses": 0})

        for trade in trades:
            days = trade.days_out
            if trade.outcome == "WIN":
                days_stats[days]["wins"] += 1
            else:
                days_stats[days]["losses"] += 1

        for days, stats in sorted(days_stats.items()):
            total = stats["wins"] + stats["losses"]
            if total == 0:
                continue

            win_rate = stats["wins"] / total
            day_label = "Same-day" if days == 0 else f"{days}-day out"

            if days > 0 and win_rate < 0.40:
                recommendation = f"Consider avoiding {day_label} trades or requiring higher edge"
            elif days == 0 and win_rate > 0.60:
                recommendation = "Same-day trades performing well - prioritize these"
            else:
                recommendation = None

            insights.append(PatternInsight(
                category="timing",
                finding=f"{day_label}: {stats['wins']}W/{stats['losses']}L ({win_rate:.0%} win rate)",
                winners=stats["wins"],
                losers=stats["losses"],
                win_rate=win_rate,
                recommendation=recommendation,
                priority="medium" if recommendation else "low"
            ))

        return insights

    def analyze_by_market_type(self, trades: List[TradeRecord]) -> List[PatternInsight]:
        """Analyze win rate by market type (threshold vs bracket)."""
        insights = []

        threshold_wins = sum(1 for t in trades if not t.is_bracket and t.outcome == "WIN")
        threshold_losses = sum(1 for t in trades if not t.is_bracket and t.outcome == "LOSS")
        bracket_wins = sum(1 for t in trades if t.is_bracket and t.outcome == "WIN")
        bracket_losses = sum(1 for t in trades if t.is_bracket and t.outcome == "LOSS")

        if threshold_wins + threshold_losses > 0:
            win_rate = threshold_wins / (threshold_wins + threshold_losses)
            insights.append(PatternInsight(
                category="market_type",
                finding=f"Threshold markets (T): {threshold_wins}W/{threshold_losses}L ({win_rate:.0%})",
                winners=threshold_wins,
                losers=threshold_losses,
                win_rate=win_rate,
                priority="medium"
            ))

        if bracket_wins + bracket_losses > 0:
            win_rate = bracket_wins / (bracket_wins + bracket_losses)
            recommendation = None
            if win_rate < 0.35:
                recommendation = "Bracket markets underperforming - consider avoiding or reducing size"
            insights.append(PatternInsight(
                category="market_type",
                finding=f"Bracket markets (B): {bracket_wins}W/{bracket_losses}L ({win_rate:.0%})",
                winners=bracket_wins,
                losers=bracket_losses,
                win_rate=win_rate,
                recommendation=recommendation,
                priority="high" if recommendation else "medium"
            ))

        return insights

    def analyze_by_direction(self, trades: List[TradeRecord]) -> List[PatternInsight]:
        """Analyze win rate by trade direction."""
        insights = []

        yes_wins = sum(1 for t in trades if t.direction == "BUY_YES" and t.outcome == "WIN")
        yes_losses = sum(1 for t in trades if t.direction == "BUY_YES" and t.outcome == "LOSS")
        no_wins = sum(1 for t in trades if t.direction == "BUY_NO" and t.outcome == "WIN")
        no_losses = sum(1 for t in trades if t.direction == "BUY_NO" and t.outcome == "LOSS")

        if yes_wins + yes_losses > 0:
            win_rate = yes_wins / (yes_wins + yes_losses)
            insights.append(PatternInsight(
                category="direction",
                finding=f"BUY_YES trades: {yes_wins}W/{yes_losses}L ({win_rate:.0%})",
                winners=yes_wins,
                losers=yes_losses,
                win_rate=win_rate,
                priority="medium"
            ))

        if no_wins + no_losses > 0:
            win_rate = no_wins / (no_wins + no_losses)
            insights.append(PatternInsight(
                category="direction",
                finding=f"BUY_NO trades: {no_wins}W/{no_losses}L ({win_rate:.0%})",
                winners=no_wins,
                losers=no_losses,
                win_rate=win_rate,
                priority="medium"
            ))

        return insights

    def get_all_insights(self) -> List[PatternInsight]:
        """Run all analyses and return combined insights."""
        trades = self.get_all_settled_trades()

        if not trades:
            return [PatternInsight(
                category="info",
                finding="No settled trades to analyze yet",
                winners=0, losers=0, win_rate=0,
                priority="low"
            )]

        insights = []
        insights.extend(self.analyze_by_probability_bucket(trades))
        insights.extend(self.analyze_by_edge_bucket(trades))
        insights.extend(self.analyze_by_city(trades))
        insights.extend(self.analyze_by_days_out(trades))
        insights.extend(self.analyze_by_market_type(trades))
        insights.extend(self.analyze_by_direction(trades))

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda x: priority_order.get(x.priority, 2))

        return insights

    def get_recommendations(self) -> List[str]:
        """Get actionable recommendations from pattern analysis."""
        insights = self.get_all_insights()

        recommendations = []
        for insight in insights:
            if insight.recommendation and insight.priority == "high":
                recommendations.append(f"[{insight.category.upper()}] {insight.recommendation}")

        return recommendations

    def generate_report(self) -> str:
        """Generate a comprehensive pattern analysis report."""
        trades = self.get_all_settled_trades()

        lines = []
        lines.append("=" * 70)
        lines.append("PATTERN ANALYSIS REPORT - Winners vs Losers")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)

        if not trades:
            lines.append("\nNo settled paper trades to analyze yet.")
            lines.append("\nAttempting to analyze real Kalshi positions...")

            # Try to show Kalshi position analysis instead
            kalshi_analysis = self.get_kalshi_positions_analysis()
            lines.append(kalshi_analysis)

            lines.append("\n💡 TIP: After trades settle, run this command again for full pattern analysis.")
            lines.append("    Settled trades will show win rates by probability, edge, city, etc.")
            return "\n".join(lines)

        # Overall stats
        total = len(trades)
        winners = sum(1 for t in trades if t.outcome == "WIN")
        losers = total - winners
        total_pnl = sum(t.pnl for t in trades)

        lines.append(f"\nOVERALL: {winners}W/{losers}L ({winners/total:.0%} win rate), P&L: ${total_pnl:.2f}")
        lines.append("-" * 70)

        # Get all insights
        insights = self.get_all_insights()

        # Group by category
        categories = {}
        for insight in insights:
            if insight.category not in categories:
                categories[insight.category] = []
            categories[insight.category].append(insight)

        # Print each category
        category_titles = {
            "probability": "BY ENTRY PROBABILITY (our calculated probability)",
            "edge": "BY EDGE AT ENTRY",
            "city": "BY CITY",
            "timing": "BY FORECAST HORIZON",
            "market_type": "BY MARKET TYPE",
            "direction": "BY TRADE DIRECTION"
        }

        for cat, title in category_titles.items():
            if cat in categories:
                lines.append(f"\n{title}")
                lines.append("-" * 50)
                for insight in categories[cat]:
                    marker = "⚠️ " if insight.priority == "high" else "  "
                    lines.append(f"{marker}{insight.finding}")
                    if insight.recommendation:
                        lines.append(f"    → {insight.recommendation}")

        # High priority recommendations summary
        recs = self.get_recommendations()
        if recs:
            lines.append("\n" + "=" * 70)
            lines.append("🎯 HIGH PRIORITY RECOMMENDATIONS")
            lines.append("=" * 70)
            for rec in recs:
                lines.append(f"  • {rec}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    analyzer = PatternAnalyzer()
    print(analyzer.generate_report())
