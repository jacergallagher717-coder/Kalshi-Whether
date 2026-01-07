"""
Model Accuracy Tracker - Track forecast accuracy by model and city.

This module:
1. Stores predictions from each model when trades are made
2. Fetches actual temperatures after settlement
3. Calculates accuracy metrics (MAE, bias) by model and city
4. Provides data to potentially auto-adjust model weights

Key insight: If ECMWF is consistently more accurate for NYC but GFS is better
for Denver, we should weight them differently per city.
"""

import sqlite3
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os

from config.settings import DATABASE_PATH
from utils.logger import get_logger

logger = get_logger("model_tracker")

MODEL_TRACKER_DB = os.path.join(DATABASE_PATH, "model_accuracy.db")


@dataclass
class ModelPrediction:
    """A single model's prediction for a specific city/date."""
    id: int
    created_at: datetime
    city: str
    target_date: date
    model_name: str
    predicted_high: Optional[float]
    predicted_low: Optional[float]
    actual_high: Optional[float] = None
    actual_low: Optional[float] = None
    high_error: Optional[float] = None
    low_error: Optional[float] = None


@dataclass
class ModelAccuracy:
    """Accuracy metrics for a model."""
    model_name: str
    total_predictions: int
    mae_high: float  # Mean Absolute Error for high temps
    mae_low: float   # Mean Absolute Error for low temps
    bias_high: float  # Positive = model runs hot, Negative = runs cold
    bias_low: float
    cities_tracked: List[str]


class ModelTracker:
    """
    Tracks model prediction accuracy over time.

    Usage:
        tracker = ModelTracker()

        # When making a trade, store predictions
        tracker.store_predictions("NYC", date(2026, 1, 7), {
            "ecmwf": {"high": 45, "low": 32},
            "gfs": {"high": 44, "low": 31},
            ...
        })

        # After settlement, update with actuals
        tracker.update_actuals("NYC", date(2026, 1, 7), actual_high=46, actual_low=33)

        # Get accuracy report
        report = tracker.get_accuracy_report()
    """

    def __init__(self, db_path: str = None):
        """Initialize the tracker."""
        self.db_path = db_path or MODEL_TRACKER_DB
        self._init_database()
        logger.info("Model accuracy tracker initialized")

    def _init_database(self):
        """Create database tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at DATETIME NOT NULL,
                city TEXT NOT NULL,
                target_date DATE NOT NULL,
                model_name TEXT NOT NULL,
                predicted_high REAL,
                predicted_low REAL,
                actual_high REAL,
                actual_low REAL,
                high_error REAL,
                low_error REAL,
                UNIQUE(city, target_date, model_name)
            )
        ''')

        # Index for fast lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_city_date
            ON predictions(city, target_date)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_model
            ON predictions(model_name)
        ''')

        conn.commit()
        conn.close()

    def store_predictions(
        self,
        city: str,
        target_date: date,
        forecasts: Dict[str, Dict[str, float]]
    ):
        """
        Store predictions from all models for a city/date.

        Args:
            city: City code (NYC, CHI, etc.)
            target_date: Date the forecast is for
            forecasts: Dict of model_name -> {"high": temp, "low": temp}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()

        for model_name, temps in forecasts.items():
            if temps is None:
                continue

            predicted_high = temps.get("high")
            predicted_low = temps.get("low")

            if predicted_high is None and predicted_low is None:
                continue

            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO predictions
                    (created_at, city, target_date, model_name, predicted_high, predicted_low)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (now, city, target_date.isoformat(), model_name, predicted_high, predicted_low))
            except Exception as e:
                logger.warning(f"Failed to store prediction for {model_name}/{city}: {e}")

        conn.commit()
        conn.close()

        logger.debug(f"Stored {len(forecasts)} model predictions for {city} on {target_date}")

    def update_actuals(
        self,
        city: str,
        target_date: date,
        actual_high: float = None,
        actual_low: float = None
    ):
        """
        Update predictions with actual observed temperatures.

        Args:
            city: City code
            target_date: Date to update
            actual_high: Actual high temperature
            actual_low: Actual low temperature
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update all predictions for this city/date
        cursor.execute('''
            UPDATE predictions
            SET actual_high = ?,
                actual_low = ?,
                high_error = CASE WHEN predicted_high IS NOT NULL AND ? IS NOT NULL
                             THEN predicted_high - ? ELSE NULL END,
                low_error = CASE WHEN predicted_low IS NOT NULL AND ? IS NOT NULL
                            THEN predicted_low - ? ELSE NULL END
            WHERE city = ? AND target_date = ?
        ''', (actual_high, actual_low, actual_high, actual_high,
              actual_low, actual_low, city, target_date.isoformat()))

        rows_updated = cursor.rowcount
        conn.commit()
        conn.close()

        if rows_updated > 0:
            logger.info(f"Updated {rows_updated} predictions with actuals for {city} on {target_date}")

        return rows_updated

    def get_model_accuracy(self, model_name: str = None, city: str = None) -> List[ModelAccuracy]:
        """
        Get accuracy metrics, optionally filtered by model or city.

        Args:
            model_name: Filter to specific model
            city: Filter to specific city

        Returns:
            List of ModelAccuracy objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        query = '''
            SELECT
                model_name,
                COUNT(*) as total,
                AVG(ABS(high_error)) as mae_high,
                AVG(ABS(low_error)) as mae_low,
                AVG(high_error) as bias_high,
                AVG(low_error) as bias_low,
                GROUP_CONCAT(DISTINCT city) as cities
            FROM predictions
            WHERE actual_high IS NOT NULL OR actual_low IS NOT NULL
        '''

        params = []
        if model_name:
            query += ' AND model_name = ?'
            params.append(model_name)
        if city:
            query += ' AND city = ?'
            params.append(city)

        query += ' GROUP BY model_name ORDER BY mae_high ASC'

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append(ModelAccuracy(
                model_name=row[0],
                total_predictions=row[1],
                mae_high=row[2] or 0.0,
                mae_low=row[3] or 0.0,
                bias_high=row[4] or 0.0,
                bias_low=row[5] or 0.0,
                cities_tracked=row[6].split(',') if row[6] else []
            ))

        return results

    def get_accuracy_by_city(self) -> Dict[str, Dict[str, float]]:
        """
        Get model accuracy broken down by city.

        Returns:
            Dict of city -> {model_name: mae}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                city,
                model_name,
                AVG(ABS(high_error)) as mae_high
            FROM predictions
            WHERE actual_high IS NOT NULL
            GROUP BY city, model_name
            ORDER BY city, mae_high ASC
        ''')

        rows = cursor.fetchall()
        conn.close()

        results = {}
        for city, model, mae in rows:
            if city not in results:
                results[city] = {}
            results[city][model] = mae

        return results

    def get_recommended_weights(self, min_predictions: int = 10) -> Dict[str, float]:
        """
        Calculate recommended model weights based on accuracy.

        Models with lower MAE get higher weights.

        Args:
            min_predictions: Minimum predictions required to include model

        Returns:
            Dict of model_name -> recommended weight (0-1, sums to 1)
        """
        accuracies = self.get_model_accuracy()

        # Filter to models with enough data
        valid = [a for a in accuracies if a.total_predictions >= min_predictions]

        if not valid:
            logger.warning("Not enough data to calculate recommended weights")
            return {}

        # Convert MAE to weights (lower MAE = higher weight)
        # Use inverse MAE, normalized
        inverse_maes = []
        for acc in valid:
            mae = acc.mae_high if acc.mae_high > 0 else 1.0
            inverse_maes.append(1.0 / mae)

        total_inverse = sum(inverse_maes)

        weights = {}
        for acc, inv_mae in zip(valid, inverse_maes):
            weights[acc.model_name] = inv_mae / total_inverse

        return weights

    def generate_report(self) -> str:
        """Generate a text report of model accuracy."""
        lines = []
        lines.append("=" * 60)
        lines.append("MODEL ACCURACY REPORT")
        lines.append("=" * 60)

        accuracies = self.get_model_accuracy()

        if not accuracies:
            lines.append("\nNo accuracy data yet. Need settled trades with actuals.")
            return "\n".join(lines)

        lines.append("\nOVERALL MODEL ACCURACY (by MAE):")
        lines.append("-" * 40)
        lines.append(f"{'Model':<15} {'Predictions':<12} {'MAE High':<10} {'Bias':<10}")
        lines.append("-" * 40)

        for acc in accuracies:
            bias_str = f"{acc.bias_high:+.1f}°F" if acc.bias_high else "N/A"
            lines.append(
                f"{acc.model_name:<15} {acc.total_predictions:<12} "
                f"{acc.mae_high:.1f}°F{'':<5} {bias_str}"
            )

        # By city breakdown
        city_acc = self.get_accuracy_by_city()
        if city_acc:
            lines.append("\n\nACCURACY BY CITY:")
            lines.append("-" * 40)
            for city, models in city_acc.items():
                lines.append(f"\n{city}:")
                for model, mae in sorted(models.items(), key=lambda x: x[1]):
                    lines.append(f"  {model}: {mae:.1f}°F MAE")

        # Recommended weights
        weights = self.get_recommended_weights()
        if weights:
            lines.append("\n\nRECOMMENDED WEIGHTS (based on accuracy):")
            lines.append("-" * 40)
            for model, weight in sorted(weights.items(), key=lambda x: -x[1]):
                lines.append(f"  {model}: {weight:.0%}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    tracker = ModelTracker()

    # Example usage
    print("Testing Model Tracker...")

    # Store some test predictions
    tracker.store_predictions("NYC", date.today(), {
        "ecmwf": {"high": 45, "low": 32},
        "gfs": {"high": 44, "low": 31},
        "nws": {"high": 46, "low": 33}
    })

    # Simulate actuals
    tracker.update_actuals("NYC", date.today(), actual_high=45, actual_low=32)

    # Generate report
    print(tracker.generate_report())
