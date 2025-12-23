"""
Data manager for coordinating data collection and storage.

Responsibilities:
1. Coordinate data collection from all sources
2. Store data in SQLite databases
3. Retrieve data for analysis and trading
4. Handle data quality issues
"""

import sqlite3
import json
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path

from config.settings import (
    FORECASTS_DB, TRADES_DB, DATABASE_PATH,
    KALSHI_PROD_URL, KALSHI_DEMO_URL, KALSHI_USE_DEMO
)
from config.locations import ACTIVE_LOCATIONS
from utils.logger import get_logger
from .kalshi_client import KalshiClient, Market, OrderBook
from .weather_client import WeatherClient, Forecast, ActualWeather

logger = get_logger("data_manager")


class DataManager:
    """
    Manages data collection, storage, and retrieval for the trading system.

    Uses production API for market data (real prices) and demo API for trading.
    """

    def __init__(self, use_production_data: bool = True):
        """
        Initialize the data manager.

        Args:
            use_production_data: If True, fetch market data from production API
                                 for real prices, while using demo for trading.
        """
        # Production client for market data (real prices, no auth needed for public data)
        if use_production_data:
            self.kalshi_client = KalshiClient(
                base_url=KALSHI_PROD_URL,
                private_key_path=None  # No auth needed for public market data
            )
            logger.info("Using PRODUCTION API for market data (real prices)")
        else:
            self.kalshi_client = KalshiClient()

        # Demo client for trade execution (with auth)
        self.trading_client = KalshiClient(
            base_url=KALSHI_DEMO_URL
        )

        self.weather_client = WeatherClient()

        # Ensure database directory exists
        Path(DATABASE_PATH).mkdir(parents=True, exist_ok=True)

        # Initialize databases
        self._init_forecasts_db()
        self._init_trades_db()

        logger.info("Data manager initialized")

    def _init_forecasts_db(self):
        """Initialize the forecasts database schema."""
        conn = sqlite3.connect(FORECASTS_DB)
        cursor = conn.cursor()

        # Weather forecasts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                location TEXT NOT NULL,
                forecast_time DATETIME NOT NULL,
                target_date DATE NOT NULL,
                high_temp_f REAL,
                low_temp_f REAL,
                raw_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_forecasts_lookup
            ON forecasts(location, target_date, source)
        ''')

        # Actual temperatures table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS actuals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT NOT NULL,
                date DATE NOT NULL,
                high_temp_f REAL,
                low_temp_f REAL,
                source TEXT DEFAULT 'nws',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(location, date)
            )
        ''')

        # Market snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                snapshot_time DATETIME NOT NULL,
                yes_price REAL,
                no_price REAL,
                yes_bid REAL,
                yes_ask REAL,
                volume INTEGER,
                open_interest INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_snapshots_ticker
            ON market_snapshots(ticker, snapshot_time)
        ''')

        conn.commit()
        conn.close()
        logger.debug("Forecasts database initialized")

    def _init_trades_db(self):
        """Initialize the trades database schema."""
        conn = sqlite3.connect(TRADES_DB)
        cursor = conn.cursor()

        # Paper trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id TEXT PRIMARY KEY,
                created_at DATETIME NOT NULL,
                ticker TEXT NOT NULL,
                location TEXT NOT NULL,
                target_date DATE NOT NULL,
                temp_threshold REAL NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                contracts INTEGER NOT NULL,
                simulated_fees REAL NOT NULL,
                our_probability REAL NOT NULL,
                edge_at_entry REAL NOT NULL,
                confidence TEXT NOT NULL,
                reasoning TEXT,
                status TEXT DEFAULT 'OPEN',
                settlement_outcome TEXT,
                actual_temp REAL,
                exit_price REAL,
                gross_pnl REAL,
                net_pnl REAL,
                settled_at DATETIME
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_status
            ON paper_trades(status)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_date
            ON paper_trades(target_date)
        ''')

        # Trade signals table (all signals, not just executed)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at DATETIME NOT NULL,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                market_price REAL NOT NULL,
                our_probability REAL NOT NULL,
                edge REAL NOT NULL,
                confidence TEXT NOT NULL,
                was_executed INTEGER DEFAULT 0,
                execution_reason TEXT
            )
        ''')

        conn.commit()
        conn.close()
        logger.debug("Trades database initialized")

    def collect_all_data(self, locations: List[str] = None) -> Dict[str, Any]:
        """
        Collect data from all sources for specified locations.

        Args:
            locations: List of location keys. Defaults to ACTIVE_LOCATIONS.

        Returns:
            Summary of collected data
        """
        locations = locations or ACTIVE_LOCATIONS
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "locations": {},
            "errors": []
        }

        for location in locations:
            location_summary = {
                "forecasts": 0,
                "markets": 0,
                "errors": []
            }

            # Collect weather forecasts
            try:
                forecasts = self.weather_client.get_all_forecasts(location)
                for source, source_forecasts in forecasts.items():
                    for forecast in source_forecasts:
                        self.store_forecast(forecast)
                        location_summary["forecasts"] += 1
            except Exception as e:
                error_msg = f"Error collecting forecasts for {location}: {e}"
                logger.error(error_msg)
                location_summary["errors"].append(error_msg)

            # Collect market data
            try:
                markets = self.kalshi_client.get_weather_markets(location)
                for market in markets:
                    self.store_market_snapshot(market)
                    location_summary["markets"] += 1
            except Exception as e:
                error_msg = f"Error collecting markets for {location}: {e}"
                logger.error(error_msg)
                location_summary["errors"].append(error_msg)

            summary["locations"][location] = location_summary

        total_forecasts = sum(loc["forecasts"] for loc in summary["locations"].values())
        total_markets = sum(loc["markets"] for loc in summary["locations"].values())
        logger.info(f"Data collection complete: {total_forecasts} forecasts, {total_markets} markets")

        return summary

    def store_forecast(self, forecast: Forecast):
        """Store a weather forecast in the database."""
        conn = sqlite3.connect(FORECASTS_DB)
        cursor = conn.cursor()

        raw_data = json.dumps(forecast.raw_data) if forecast.raw_data else None

        cursor.execute('''
            INSERT INTO forecasts (source, location, forecast_time, target_date,
                                   high_temp_f, low_temp_f, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            forecast.source,
            forecast.location,
            forecast.forecast_time.isoformat(),
            forecast.target_date.isoformat(),
            forecast.high_temp_f,
            forecast.low_temp_f,
            raw_data
        ))

        conn.commit()
        conn.close()

    def store_market_snapshot(self, market: Market):
        """Store a market snapshot in the database."""
        conn = sqlite3.connect(FORECASTS_DB)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO market_snapshots (ticker, snapshot_time, yes_price, no_price,
                                          yes_bid, yes_ask, volume, open_interest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            market.ticker,
            datetime.utcnow().isoformat(),
            market.yes_price,
            market.no_price,
            market.yes_bid,
            market.yes_ask,
            market.volume,
            market.open_interest
        ))

        conn.commit()
        conn.close()

    def store_actual(self, actual: ActualWeather):
        """Store actual weather data (for settlement)."""
        conn = sqlite3.connect(FORECASTS_DB)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO actuals (location, date, high_temp_f, low_temp_f, source)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            actual.location,
            actual.date.isoformat(),
            actual.high_temp_f,
            actual.low_temp_f,
            actual.source
        ))

        conn.commit()
        conn.close()

    def get_latest_forecasts(self, location: str, target_date: date) -> Dict[str, Forecast]:
        """
        Get the most recent forecasts for a specific date from all sources.

        Args:
            location: Location key
            target_date: Date to get forecasts for

        Returns:
            Dictionary mapping source to Forecast
        """
        conn = sqlite3.connect(FORECASTS_DB)
        cursor = conn.cursor()

        # Get latest forecast from each source
        cursor.execute('''
            SELECT source, forecast_time, high_temp_f, low_temp_f
            FROM forecasts
            WHERE location = ? AND target_date = ?
            ORDER BY forecast_time DESC
        ''', (location, target_date.isoformat()))

        rows = cursor.fetchall()
        conn.close()

        # Group by source, keeping only most recent
        forecasts = {}
        seen_sources = set()

        for row in rows:
            source = row[0]
            if source not in seen_sources:
                forecasts[source] = Forecast(
                    source=source,
                    location=location,
                    forecast_time=datetime.fromisoformat(row[1]),
                    target_date=target_date,
                    high_temp_f=row[2],
                    low_temp_f=row[3]
                )
                seen_sources.add(source)

        return forecasts

    def get_actual(self, location: str, target_date: date) -> Optional[ActualWeather]:
        """
        Get actual weather for a specific date.

        Args:
            location: Location key
            target_date: Date to get actual for

        Returns:
            ActualWeather or None if not available
        """
        conn = sqlite3.connect(FORECASTS_DB)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT high_temp_f, low_temp_f, source
            FROM actuals
            WHERE location = ? AND date = ?
        ''', (location, target_date.isoformat()))

        row = cursor.fetchone()
        conn.close()

        if row:
            return ActualWeather(
                location=location,
                date=target_date,
                high_temp_f=row[0],
                low_temp_f=row[1],
                source=row[2]
            )

        # Try to fetch from API if not in database
        actual = self.weather_client.get_actual_for_date(location, target_date)
        if actual:
            self.store_actual(actual)
            return actual

        return None

    def get_markets(self, location: str = None) -> List[Market]:
        """
        Get current weather markets.

        Args:
            location: Optional location filter

        Returns:
            List of Market objects
        """
        return self.kalshi_client.get_weather_markets(location)

    def get_market_history(self, ticker: str,
                          hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Get historical market snapshots.

        Args:
            ticker: Market ticker
            hours_back: Number of hours of history to retrieve

        Returns:
            List of snapshot dictionaries
        """
        conn = sqlite3.connect(FORECASTS_DB)
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()

        cursor.execute('''
            SELECT snapshot_time, yes_price, no_price, volume
            FROM market_snapshots
            WHERE ticker = ? AND snapshot_time > ?
            ORDER BY snapshot_time DESC
        ''', (ticker, cutoff))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "timestamp": row[0],
                "yes_price": row[1],
                "no_price": row[2],
                "volume": row[3]
            }
            for row in rows
        ]

    def test_connections(self) -> Dict[str, bool]:
        """
        Test connectivity to all data sources.

        Returns:
            Dictionary mapping source names to connection status
        """
        results = {}

        # Test Kalshi production (market data)
        results["kalshi_data"] = self.kalshi_client.test_connection()

        # Test Kalshi demo (trading) - requires login
        results["kalshi_trading"] = self.trading_client.login()

        # Test weather APIs
        weather_status = self.weather_client.test_connection()
        results.update(weather_status)

        return results


# Example usage
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    manager = DataManager()

    # Test connections
    print("Testing connections...")
    status = manager.test_connections()
    for source, connected in status.items():
        print(f"  {source}: {'✓' if connected else '✗'}")

    # Collect data
    print("\nCollecting data...")
    summary = manager.collect_all_data()
    print(f"Summary: {json.dumps(summary, indent=2)}")
