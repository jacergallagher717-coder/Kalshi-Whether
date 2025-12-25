"""
Trade analytics and backtesting data collection.

Logs detailed prediction data for model improvement:
- Weather forecasts from each source
- Our predicted probability vs actual outcome
- Market price at prediction time
- Settlement results
"""

import sqlite3
import json
from datetime import datetime, date
from typing import Dict, List, Optional
from pathlib import Path

from config.settings import DATA_DIR
from utils.logger import get_logger

logger = get_logger("analytics")

ANALYTICS_DB = DATA_DIR / "analytics.db"


class TradingAnalytics:
    """Collects and analyzes trading data for model improvement."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or ANALYTICS_DB
        self._init_database()

    def _init_database(self):
        """Create analytics tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Predictions table - logs every prediction made
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                ticker TEXT NOT NULL,
                location TEXT NOT NULL,
                target_date DATE NOT NULL,
                market_type TEXT NOT NULL,
                temp_threshold REAL NOT NULL,

                -- Market data at prediction time
                market_yes_price REAL NOT NULL,
                market_implied_prob REAL NOT NULL,

                -- Our predictions
                our_probability REAL NOT NULL,
                edge REAL NOT NULL,
                confidence TEXT,

                -- Individual weather source forecasts
                ecmwf_temp REAL,
                gfs_temp REAL,
                nws_temp REAL,
                visualcrossing_temp REAL,
                ensemble_temp REAL,
                ensemble_std REAL,

                -- Action taken
                action_taken TEXT,  -- 'BUY_YES', 'BUY_NO', 'SKIP'
                contracts INTEGER,
                entry_price REAL,

                -- Outcome (filled after settlement)
                actual_temp REAL,
                outcome TEXT,  -- 'YES_WIN', 'NO_WIN', 'PENDING'
                prediction_correct INTEGER,  -- 1 or 0
                pnl REAL,

                -- Metadata
                weather_data_json TEXT
            )
        ''')

        # Model performance by source
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS source_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                location TEXT NOT NULL,
                source TEXT NOT NULL,
                predicted_temp REAL NOT NULL,
                actual_temp REAL,
                error REAL,
                abs_error REAL
            )
        ''')

        # Daily summary
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date DATE PRIMARY KEY,
                total_predictions INTEGER,
                trades_executed INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                total_pnl REAL,
                avg_edge REAL,
                avg_edge_when_correct REAL,
                avg_edge_when_wrong REAL
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Analytics database initialized")

    def log_prediction(
        self,
        ticker: str,
        location: str,
        target_date: date,
        market_type: str,
        temp_threshold: float,
        market_yes_price: float,
        our_probability: float,
        edge: float,
        weather_forecasts: Dict,
        action: str = None,
        contracts: int = 0,
        entry_price: float = 0,
        confidence: str = None
    ):
        """Log a prediction for later analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO predictions (
                timestamp, ticker, location, target_date, market_type, temp_threshold,
                market_yes_price, market_implied_prob, our_probability, edge, confidence,
                ecmwf_temp, gfs_temp, nws_temp, visualcrossing_temp,
                ensemble_temp, ensemble_std,
                action_taken, contracts, entry_price, outcome,
                weather_data_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            ticker, location, target_date.isoformat(), market_type, temp_threshold,
            market_yes_price, market_yes_price,  # implied prob = price for binary
            our_probability, edge, confidence,
            weather_forecasts.get('ecmwf'),
            weather_forecasts.get('gfs'),
            weather_forecasts.get('nws'),
            weather_forecasts.get('visualcrossing'),
            weather_forecasts.get('ensemble'),
            weather_forecasts.get('std'),
            action, contracts, entry_price, 'PENDING',
            json.dumps(weather_forecasts)
        ))

        conn.commit()
        conn.close()
        logger.debug(f"Logged prediction for {ticker}")

    def update_outcome(
        self,
        ticker: str,
        target_date: date,
        actual_temp: float,
        outcome: str,
        pnl: float = None
    ):
        """Update prediction with actual outcome after settlement."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Determine if prediction was correct
        cursor.execute('''
            SELECT id, action_taken, our_probability FROM predictions
            WHERE ticker = ? AND target_date = ? AND outcome = 'PENDING'
        ''', (ticker, target_date.isoformat()))

        row = cursor.fetchone()
        if row:
            pred_id, action, our_prob = row

            # Was our prediction correct?
            if action == 'BUY_YES':
                correct = 1 if outcome == 'YES_WIN' else 0
            elif action == 'BUY_NO':
                correct = 1 if outcome == 'NO_WIN' else 0
            else:
                correct = None

            cursor.execute('''
                UPDATE predictions
                SET actual_temp = ?, outcome = ?, prediction_correct = ?, pnl = ?
                WHERE id = ?
            ''', (actual_temp, outcome, correct, pnl, pred_id))

        conn.commit()
        conn.close()

    def get_model_accuracy(self, days: int = 30) -> Dict:
        """Get model accuracy statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN prediction_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(edge) as avg_edge,
                AVG(CASE WHEN prediction_correct = 1 THEN edge END) as edge_when_correct,
                AVG(CASE WHEN prediction_correct = 0 THEN edge END) as edge_when_wrong,
                SUM(pnl) as total_pnl
            FROM predictions
            WHERE outcome != 'PENDING'
            AND timestamp > datetime('now', ?)
        ''', (f'-{days} days',))

        row = cursor.fetchone()
        conn.close()

        if row and row[0] > 0:
            return {
                'total_predictions': row[0],
                'correct': row[1],
                'win_rate': row[1] / row[0] if row[0] > 0 else 0,
                'avg_edge': row[2] or 0,
                'edge_when_correct': row[3] or 0,
                'edge_when_wrong': row[4] or 0,
                'total_pnl': row[5] or 0
            }
        return {'total_predictions': 0}

    def get_source_accuracy(self, days: int = 30) -> Dict:
        """Get accuracy by weather source."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        sources = ['ecmwf', 'gfs', 'nws', 'visualcrossing']
        results = {}

        for source in sources:
            cursor.execute(f'''
                SELECT
                    AVG(ABS({source}_temp - actual_temp)) as mae,
                    COUNT(*) as count
                FROM predictions
                WHERE {source}_temp IS NOT NULL
                AND actual_temp IS NOT NULL
                AND timestamp > datetime('now', ?)
            ''', (f'-{days} days',))

            row = cursor.fetchone()
            if row and row[1] > 0:
                results[source] = {
                    'mean_absolute_error': row[0],
                    'sample_count': row[1]
                }

        conn.close()
        return results

    def get_edge_vs_outcome(self) -> List[Dict]:
        """Get edge buckets vs actual win rate for calibration analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                CASE
                    WHEN edge < 0.1 THEN '0-10%'
                    WHEN edge < 0.2 THEN '10-20%'
                    WHEN edge < 0.3 THEN '20-30%'
                    WHEN edge < 0.4 THEN '30-40%'
                    ELSE '40%+'
                END as edge_bucket,
                COUNT(*) as total,
                SUM(CASE WHEN prediction_correct = 1 THEN 1 ELSE 0 END) as wins,
                AVG(edge) as avg_edge
            FROM predictions
            WHERE outcome != 'PENDING'
            GROUP BY edge_bucket
            ORDER BY avg_edge
        ''')

        results = []
        for row in cursor.fetchall():
            results.append({
                'edge_bucket': row[0],
                'total': row[1],
                'wins': row[2],
                'actual_win_rate': row[2] / row[1] if row[1] > 0 else 0,
                'avg_edge': row[3]
            })

        conn.close()
        return results

    def print_report(self, days: int = 30):
        """Print a comprehensive analytics report."""
        print("\n" + "=" * 60)
        print(f"TRADING ANALYTICS REPORT (Last {days} days)")
        print("=" * 60)

        # Overall accuracy
        accuracy = self.get_model_accuracy(days)
        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"   Total predictions: {accuracy.get('total_predictions', 0)}")
        print(f"   Win rate: {accuracy.get('win_rate', 0):.1%}")
        print(f"   Average edge: {accuracy.get('avg_edge', 0):.1%}")
        print(f"   Total P&L: ${accuracy.get('total_pnl', 0):.2f}")

        # Edge calibration
        print(f"\n📈 EDGE CALIBRATION (Predicted Edge vs Actual Win Rate)")
        edge_data = self.get_edge_vs_outcome()
        for bucket in edge_data:
            print(f"   {bucket['edge_bucket']}: {bucket['actual_win_rate']:.1%} actual "
                  f"(n={bucket['total']})")

        # Source accuracy
        print(f"\n🌤️ WEATHER SOURCE ACCURACY (Mean Absolute Error)")
        source_data = self.get_source_accuracy(days)
        for source, data in source_data.items():
            print(f"   {source}: {data['mean_absolute_error']:.1f}°F error "
                  f"(n={data['sample_count']})")

        print("\n" + "=" * 60)


# CLI for viewing analytics
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trading Analytics")
    parser.add_argument("--days", type=int, default=30, help="Days to analyze")
    args = parser.parse_args()

    analytics = TradingAnalytics()
    analytics.print_report(args.days)
