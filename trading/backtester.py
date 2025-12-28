"""
Backtesting system for weather trading model validation.

This module:
1. Fetches historical actual temperatures from NWS
2. Compares our model's predictions vs actual outcomes
3. Simulates trading performance over historical data
4. Generates reports on model accuracy and profitability
"""

import sqlite3
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics

from config.settings import DATABASE_PATH
from config.locations import LOCATIONS
from utils.logger import get_logger

logger = get_logger("backtester")

BACKTEST_DB = DATABASE_PATH + "backtest.db"


@dataclass
class HistoricalDay:
    """Historical weather data for a single day."""
    date: date
    location: str
    actual_high: float
    actual_low: float
    forecast_high: Optional[float] = None
    forecast_low: Optional[float] = None
    forecast_error: Optional[float] = None


@dataclass
class BacktestTrade:
    """Simulated trade for backtesting."""
    date: date
    location: str
    market_type: str  # "high" or "low"
    threshold: float
    direction: str  # "BUY_YES" or "BUY_NO"
    entry_price: float
    our_probability: float
    edge: float
    actual_temp: float
    outcome: str  # "WIN" or "LOSS"
    pnl: float


class WeatherBacktester:
    """
    Backtesting system for validating weather prediction model.

    Uses historical NWS data to:
    - Validate our temperature forecasts vs actuals
    - Simulate what trades we would have made
    - Calculate theoretical P&L
    """

    # NWS Climate Data Online API
    NWS_HISTORY_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

    # Station IDs for each location
    STATION_MAP = {
        "NYC": "USW00094728",  # Central Park
        "CHI": "USW00094846",  # O'Hare
        "LA": "USW00023174",   # LAX
        "MIA": "USW00012839",  # Miami Airport
        "AUS": "USW00013904",  # Austin
        "DEN": "USW00023062",  # Denver
        "PHI": "USW00013739",  # Philadelphia
    }

    def __init__(self, db_path: str = None):
        self.db_path = db_path or BACKTEST_DB
        self._init_database()

    def _init_database(self):
        """Create backtesting tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_weather (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                location TEXT NOT NULL,
                actual_high REAL NOT NULL,
                actual_low REAL NOT NULL,
                source TEXT DEFAULT 'NWS',
                UNIQUE(date, location)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date DATETIME NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                location TEXT,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                total_pnl REAL,
                avg_edge REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                config_json TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                date DATE NOT NULL,
                location TEXT NOT NULL,
                market_type TEXT NOT NULL,
                threshold REAL NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                our_probability REAL NOT NULL,
                edge REAL NOT NULL,
                actual_temp REAL NOT NULL,
                outcome TEXT NOT NULL,
                pnl REAL NOT NULL,
                FOREIGN KEY(run_id) REFERENCES backtest_results(id)
            )
        ''')

        conn.commit()
        conn.close()

    def fetch_historical_temps(
        self,
        location: str,
        start_date: date,
        end_date: date
    ) -> List[HistoricalDay]:
        """
        Fetch historical temperatures from NWS Climate Data.

        Args:
            location: Location code (e.g., "NYC")
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of HistoricalDay objects
        """
        station_id = self.STATION_MAP.get(location)
        if not station_id:
            logger.error(f"Unknown location: {location}")
            return []

        # Try to get from database first
        cached = self._get_cached_temps(location, start_date, end_date)
        if cached:
            return cached

        # Fetch from NWS NCEI API
        try:
            params = {
                "dataset": "daily-summaries",
                "stations": station_id,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "dataTypes": "TMAX,TMIN",
                "units": "standard",
                "format": "json"
            }

            response = requests.get(self.NWS_HISTORY_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = []
            for record in data:
                try:
                    record_date = datetime.strptime(record["DATE"], "%Y-%m-%d").date()
                    # TMAX/TMIN are in tenths of degrees Celsius from NCEI
                    # Convert to Fahrenheit
                    tmax_c = record.get("TMAX", 0) / 10
                    tmin_c = record.get("TMIN", 0) / 10
                    tmax_f = tmax_c * 9/5 + 32
                    tmin_f = tmin_c * 9/5 + 32

                    day = HistoricalDay(
                        date=record_date,
                        location=location,
                        actual_high=tmax_f,
                        actual_low=tmin_f
                    )
                    results.append(day)

                    # Cache to database
                    self._cache_temp(day)

                except (KeyError, ValueError) as e:
                    logger.debug(f"Skipping record: {e}")
                    continue

            logger.info(f"Fetched {len(results)} days of history for {location}")
            return results

        except Exception as e:
            logger.error(f"Error fetching historical temps: {e}")
            return []

    def _get_cached_temps(
        self,
        location: str,
        start_date: date,
        end_date: date
    ) -> List[HistoricalDay]:
        """Get cached temperatures from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT date, location, actual_high, actual_low
            FROM historical_weather
            WHERE location = ? AND date >= ? AND date <= ?
            ORDER BY date
        ''', (location, start_date.isoformat(), end_date.isoformat()))

        results = []
        for row in cursor.fetchall():
            results.append(HistoricalDay(
                date=datetime.strptime(row[0], "%Y-%m-%d").date(),
                location=row[1],
                actual_high=row[2],
                actual_low=row[3]
            ))

        conn.close()

        # Only return if we have complete data
        expected_days = (end_date - start_date).days + 1
        if len(results) >= expected_days * 0.9:  # Allow 10% missing
            return results
        return []

    def _cache_temp(self, day: HistoricalDay):
        """Cache a temperature record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO historical_weather
            (date, location, actual_high, actual_low)
            VALUES (?, ?, ?, ?)
        ''', (day.date.isoformat(), day.location, day.actual_high, day.actual_low))

        conn.commit()
        conn.close()

    def simulate_trade(
        self,
        day: HistoricalDay,
        market_type: str,
        threshold: float,
        market_price: float,
        our_probability: float,
        min_edge: float = 0.10
    ) -> Optional[BacktestTrade]:
        """
        Simulate a trade for backtesting.

        Args:
            day: Historical day data
            market_type: "high" or "low"
            threshold: Temperature threshold
            market_price: Simulated market price (YES price)
            our_probability: Our model's probability
            min_edge: Minimum edge to trade

        Returns:
            BacktestTrade if we would have traded, None otherwise
        """
        actual_temp = day.actual_high if market_type == "high" else day.actual_low

        # Calculate edge
        market_prob = market_price

        # Determine direction based on edge
        if our_probability > market_prob + min_edge:
            direction = "BUY_YES"
            edge = our_probability - market_prob
            entry_price = market_price
        elif our_probability < market_prob - min_edge:
            direction = "BUY_NO"
            edge = market_prob - our_probability
            entry_price = 1 - market_price
        else:
            return None  # No edge, no trade

        # Determine outcome
        if market_type == "high":
            # For threshold markets: YES wins if temp >= threshold
            actual_yes = actual_temp >= threshold
        else:
            # For low markets: YES wins if temp <= threshold
            actual_yes = actual_temp <= threshold

        if direction == "BUY_YES":
            won = actual_yes
            pnl = (1.0 - entry_price) if won else -entry_price
        else:
            won = not actual_yes
            pnl = (1.0 - entry_price) if won else -entry_price

        return BacktestTrade(
            date=day.date,
            location=day.location,
            market_type=market_type,
            threshold=threshold,
            direction=direction,
            entry_price=entry_price,
            our_probability=our_probability,
            edge=edge,
            actual_temp=actual_temp,
            outcome="WIN" if won else "LOSS",
            pnl=pnl
        )

    def run_backtest(
        self,
        location: str,
        start_date: date,
        end_date: date,
        thresholds: List[float] = None,
        min_edge: float = 0.10
    ) -> Dict:
        """
        Run a full backtest over historical data.

        Args:
            location: Location to backtest
            start_date: Start date
            end_date: End date
            thresholds: Temperature thresholds to test
            min_edge: Minimum edge to trade

        Returns:
            Backtest results dictionary
        """
        # Default thresholds based on typical NYC temps
        if thresholds is None:
            thresholds = list(range(20, 100, 5))  # 20°F to 95°F in 5° increments

        # Fetch historical data
        history = self.fetch_historical_temps(location, start_date, end_date)
        if not history:
            logger.error("No historical data available")
            return {"error": "No historical data"}

        trades = []

        for day in history:
            for threshold in thresholds:
                # Simulate market price (use historical distribution)
                # This is a simplification - real backtest would use actual market prices
                temp_diff = day.actual_high - threshold

                # Estimate what market price would have been
                # (This is approximate - ideally we'd have historical prices)
                if temp_diff > 10:
                    market_price = 0.85  # Very likely YES
                elif temp_diff > 5:
                    market_price = 0.70
                elif temp_diff > 0:
                    market_price = 0.55
                elif temp_diff > -5:
                    market_price = 0.45
                elif temp_diff > -10:
                    market_price = 0.30
                else:
                    market_price = 0.15  # Very likely NO

                # Our model's probability (based on forecast accuracy)
                # Simulate 2°F forecast error on average
                forecast_temp = day.actual_high + statistics.gauss(0, 2)
                forecast_diff = forecast_temp - threshold

                if forecast_diff > 8:
                    our_prob = 0.95
                elif forecast_diff > 4:
                    our_prob = 0.80
                elif forecast_diff > 0:
                    our_prob = 0.60
                elif forecast_diff > -4:
                    our_prob = 0.40
                elif forecast_diff > -8:
                    our_prob = 0.20
                else:
                    our_prob = 0.05

                trade = self.simulate_trade(
                    day, "high", threshold, market_price, our_prob, min_edge
                )
                if trade:
                    trades.append(trade)

        # Calculate statistics
        if not trades:
            return {"total_trades": 0, "message": "No trades met criteria"}

        wins = sum(1 for t in trades if t.outcome == "WIN")
        losses = len(trades) - wins
        total_pnl = sum(t.pnl for t in trades)
        avg_edge = statistics.mean(t.edge for t in trades)

        # Calculate Sharpe ratio (simplified)
        if len(trades) > 1:
            returns = [t.pnl for t in trades]
            sharpe = (statistics.mean(returns) / statistics.stdev(returns)) * (252 ** 0.5) if statistics.stdev(returns) > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in trades:
            cumulative += t.pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        results = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "location": location,
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(trades),
            "total_pnl": total_pnl,
            "avg_edge": avg_edge,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "avg_pnl_per_trade": total_pnl / len(trades),
            "trades": trades
        }

        # Save results
        self._save_results(results)

        return results

    def _save_results(self, results: Dict):
        """Save backtest results to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO backtest_results
            (run_date, start_date, end_date, location, total_trades, wins, losses,
             win_rate, total_pnl, avg_edge, sharpe_ratio, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            results["start_date"],
            results["end_date"],
            results["location"],
            results["total_trades"],
            results["wins"],
            results["losses"],
            results["win_rate"],
            results["total_pnl"],
            results["avg_edge"],
            results["sharpe_ratio"],
            results["max_drawdown"]
        ))

        run_id = cursor.lastrowid

        # Save individual trades
        for trade in results.get("trades", []):
            cursor.execute('''
                INSERT INTO backtest_trades
                (run_id, date, location, market_type, threshold, direction,
                 entry_price, our_probability, edge, actual_temp, outcome, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                trade.date.isoformat(),
                trade.location,
                trade.market_type,
                trade.threshold,
                trade.direction,
                trade.entry_price,
                trade.our_probability,
                trade.edge,
                trade.actual_temp,
                trade.outcome,
                trade.pnl
            ))

        conn.commit()
        conn.close()

    def print_report(self, results: Dict):
        """Print a formatted backtest report."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"Location: {results['location']}")
        print("-" * 60)
        print(f"Total Trades:     {results['total_trades']}")
        print(f"Wins:             {results['wins']}")
        print(f"Losses:           {results['losses']}")
        print(f"Win Rate:         {results['win_rate']:.1%}")
        print("-" * 60)
        print(f"Total P&L:        ${results['total_pnl']:.2f}")
        print(f"Avg P&L/Trade:    ${results['avg_pnl_per_trade']:.2f}")
        print(f"Average Edge:     {results['avg_edge']:.1%}")
        print(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:     ${results['max_drawdown']:.2f}")
        print("=" * 60)

        # Win rate by edge bucket
        trades = results.get("trades", [])
        if trades:
            print("\nWin Rate by Edge Bucket:")
            buckets = {}
            for t in trades:
                bucket = f"{int(t.edge * 100 // 10) * 10}-{int(t.edge * 100 // 10) * 10 + 10}%"
                if bucket not in buckets:
                    buckets[bucket] = {"wins": 0, "total": 0}
                buckets[bucket]["total"] += 1
                if t.outcome == "WIN":
                    buckets[bucket]["wins"] += 1

            for bucket, stats in sorted(buckets.items()):
                wr = stats["wins"] / stats["total"] if stats["total"] > 0 else 0
                print(f"  {bucket}: {wr:.1%} ({stats['total']} trades)")

        print("\n")


# CLI for running backtests
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weather Trading Backtester")
    parser.add_argument("--location", default="NYC", help="Location to backtest")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest")
    parser.add_argument("--min-edge", type=float, default=0.10, help="Minimum edge")
    args = parser.parse_args()

    backtester = WeatherBacktester()

    end_date = date.today() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=args.days)

    print(f"\nRunning backtest for {args.location}...")
    print(f"Period: {start_date} to {end_date}")
    print(f"Minimum edge: {args.min_edge:.0%}")

    results = backtester.run_backtest(
        location=args.location,
        start_date=start_date,
        end_date=end_date,
        min_edge=args.min_edge
    )

    if "error" not in results:
        backtester.print_report(results)
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")
