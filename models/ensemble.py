"""
Ensemble model for combining multiple weather forecast sources.

The ensemble model:
1. Weights forecasts based on historical accuracy
2. Adjusts for known model biases
3. Provides uncertainty estimates
4. Can be trained on historical data
"""

import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from config.settings import MODEL_WEIGHTS, TEMP_UNCERTAINTY
from utils.logger import get_logger
from .probability import forecast_to_probability, ensemble_probability

logger = get_logger("ensemble")


@dataclass
class ModelPerformance:
    """Track performance metrics for a forecast model."""
    source: str
    total_forecasts: int = 0
    sum_error: float = 0.0  # Sum of (forecast - actual)
    sum_abs_error: float = 0.0  # Sum of |forecast - actual|
    sum_squared_error: float = 0.0  # Sum of (forecast - actual)^2

    @property
    def mean_error(self) -> float:
        """Mean error (bias): positive = over-predicting."""
        if self.total_forecasts == 0:
            return 0.0
        return self.sum_error / self.total_forecasts

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        if self.total_forecasts == 0:
            return 0.0
        return self.sum_abs_error / self.total_forecasts

    @property
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        if self.total_forecasts == 0:
            return 0.0
        return np.sqrt(self.sum_squared_error / self.total_forecasts)


@dataclass
class EnsembleForecast:
    """Result of ensemble forecast calculation."""
    target_date: date
    location: str
    high_temp: Optional[float]
    low_temp: Optional[float]
    high_std: float  # Uncertainty in high temp
    low_std: float  # Uncertainty in low temp
    source_forecasts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model_agreement: float = 0.0  # 0-1, higher = more agreement
    created_at: datetime = field(default_factory=datetime.utcnow)


class EnsembleModel:
    """
    Ensemble model that combines multiple forecast sources.

    Features:
    - Weighted combination of forecasts
    - Dynamic weight adjustment based on performance
    - Bias correction for known model tendencies
    - Uncertainty quantification
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the ensemble model.

        Args:
            weights: Initial model weights. Defaults to MODEL_WEIGHTS.
        """
        self.weights = weights or MODEL_WEIGHTS.copy()
        self.performance = {}  # Track performance by source
        self.bias_corrections = {}  # Learned bias corrections

        logger.info(f"Ensemble model initialized with weights: {self.weights}")

    def combine_forecasts(
        self,
        forecasts: Dict[str, Dict[str, float]],
        target_date: date,
        location: str
    ) -> EnsembleForecast:
        """
        Combine multiple forecasts into an ensemble forecast.

        Args:
            forecasts: Dict mapping source to {"high": temp, "low": temp}
            target_date: Date being forecasted
            location: Location identifier

        Returns:
            EnsembleForecast with combined values and uncertainty
        """
        high_temps = []
        low_temps = []
        high_weights = []
        low_weights = []

        for source, temps in forecasts.items():
            weight = self.weights.get(source, 0.2)

            # Apply bias correction if available
            bias = self.bias_corrections.get(source, {})

            if temps.get("high") is not None:
                corrected_high = temps["high"] - bias.get("high", 0)
                high_temps.append(corrected_high)
                high_weights.append(weight)

            if temps.get("low") is not None:
                corrected_low = temps["low"] - bias.get("low", 0)
                low_temps.append(corrected_low)
                low_weights.append(weight)

        # Calculate weighted averages
        if high_temps:
            total_weight = sum(high_weights)
            ensemble_high = sum(t * w for t, w in zip(high_temps, high_weights)) / total_weight
            high_std = np.std(high_temps) if len(high_temps) > 1 else 2.0
        else:
            ensemble_high = None
            high_std = 0.0

        if low_temps:
            total_weight = sum(low_weights)
            ensemble_low = sum(t * w for t, w in zip(low_temps, low_weights)) / total_weight
            low_std = np.std(low_temps) if len(low_temps) > 1 else 2.0
        else:
            ensemble_low = None
            low_std = 0.0

        # Calculate model agreement (inverse of spread)
        if high_temps and len(high_temps) > 1:
            spread = np.std(high_temps)
            # Map spread to agreement: 0°F spread = 1.0 agreement, 10°F spread = 0.0
            model_agreement = max(0.0, 1.0 - spread / 10.0)
        else:
            model_agreement = 0.5  # Unknown agreement with single model

        return EnsembleForecast(
            target_date=target_date,
            location=location,
            high_temp=ensemble_high,
            low_temp=ensemble_low,
            high_std=high_std,
            low_std=low_std,
            source_forecasts=forecasts,
            model_agreement=model_agreement
        )

    def calculate_probability(
        self,
        forecasts: Dict[str, Dict[str, float]],
        threshold: float,
        days_out: int,
        temp_type: str = "high"
    ) -> Dict[str, any]:
        """
        Calculate ensemble probability of exceeding threshold.

        Args:
            forecasts: Dict mapping source to {"high": temp, "low": temp}
            threshold: Temperature threshold
            days_out: Days until settlement
            temp_type: "high" or "low"

        Returns:
            Dictionary with ensemble probability and details
        """
        # Extract relevant temperatures
        temp_forecasts = {}
        for source, temps in forecasts.items():
            temp = temps.get(temp_type)
            if temp is not None:
                # Apply bias correction
                bias = self.bias_corrections.get(source, {}).get(temp_type, 0)
                temp_forecasts[source] = temp - bias

        # Determine direction based on temp type
        direction = "above" if temp_type == "high" else "below"

        # Use probability module for ensemble calculation
        return ensemble_probability(
            temp_forecasts,
            threshold,
            days_out,
            direction,
            self.weights
        )

    def record_outcome(
        self,
        forecasts: Dict[str, Dict[str, float]],
        actual_high: float,
        actual_low: float
    ):
        """
        Record forecast vs actual outcome for performance tracking.

        Args:
            forecasts: Dict of forecasts that were made
            actual_high: Actual high temperature
            actual_low: Actual low temperature
        """
        for source, temps in forecasts.items():
            if source not in self.performance:
                self.performance[source] = {
                    "high": ModelPerformance(source=source),
                    "low": ModelPerformance(source=source)
                }

            # Record high temp performance
            if temps.get("high") is not None:
                perf = self.performance[source]["high"]
                error = temps["high"] - actual_high
                perf.total_forecasts += 1
                perf.sum_error += error
                perf.sum_abs_error += abs(error)
                perf.sum_squared_error += error ** 2

            # Record low temp performance
            if temps.get("low") is not None:
                perf = self.performance[source]["low"]
                error = temps["low"] - actual_low
                perf.total_forecasts += 1
                perf.sum_error += error
                perf.sum_abs_error += abs(error)
                perf.sum_squared_error += error ** 2

        logger.debug(f"Recorded outcome: actual high={actual_high}, low={actual_low}")

    def update_weights(self, min_samples: int = 10):
        """
        Update model weights based on observed performance.

        Uses inverse RMSE weighting - better models get higher weights.

        Args:
            min_samples: Minimum samples required before adjusting weights
        """
        rmse_scores = {}

        for source in self.weights.keys():
            if source not in self.performance:
                continue

            high_perf = self.performance[source]["high"]
            low_perf = self.performance[source]["low"]

            # Require minimum samples
            if high_perf.total_forecasts < min_samples:
                continue

            # Average RMSE across high and low
            rmse = (high_perf.rmse + low_perf.rmse) / 2
            if rmse > 0:
                rmse_scores[source] = rmse

        if not rmse_scores:
            logger.info("Not enough data to update weights")
            return

        # Calculate inverse RMSE weights
        inverse_rmse = {s: 1.0 / r for s, r in rmse_scores.items()}
        total = sum(inverse_rmse.values())

        for source in inverse_rmse:
            new_weight = inverse_rmse[source] / total
            old_weight = self.weights.get(source, 0)
            # Smooth update: 80% old weight, 20% new
            self.weights[source] = 0.8 * old_weight + 0.2 * new_weight

        logger.info(f"Updated weights: {self.weights}")

    def update_bias_corrections(self, min_samples: int = 10):
        """
        Calculate and update bias corrections for each model.

        Args:
            min_samples: Minimum samples required before calculating bias
        """
        for source in self.weights.keys():
            if source not in self.performance:
                continue

            high_perf = self.performance[source]["high"]
            low_perf = self.performance[source]["low"]

            if source not in self.bias_corrections:
                self.bias_corrections[source] = {}

            # Only apply bias correction if we have enough data
            if high_perf.total_forecasts >= min_samples:
                # Bias = mean error (positive = model over-predicts)
                self.bias_corrections[source]["high"] = high_perf.mean_error

            if low_perf.total_forecasts >= min_samples:
                self.bias_corrections[source]["low"] = low_perf.mean_error

        if self.bias_corrections:
            logger.info(f"Updated bias corrections: {self.bias_corrections}")

    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of model performance metrics.

        Returns:
            Dictionary with performance stats for each model
        """
        summary = {}

        for source, perfs in self.performance.items():
            summary[source] = {
                "high_mae": perfs["high"].mae,
                "high_rmse": perfs["high"].rmse,
                "high_bias": perfs["high"].mean_error,
                "low_mae": perfs["low"].mae,
                "low_rmse": perfs["low"].rmse,
                "low_bias": perfs["low"].mean_error,
                "total_forecasts": perfs["high"].total_forecasts
            }

        return summary

    def save_state(self, filepath: str):
        """Save model state to file."""
        state = {
            "weights": self.weights,
            "bias_corrections": self.bias_corrections,
            "performance": {
                source: {
                    temp_type: {
                        "total_forecasts": perf.total_forecasts,
                        "sum_error": perf.sum_error,
                        "sum_abs_error": perf.sum_abs_error,
                        "sum_squared_error": perf.sum_squared_error
                    }
                    for temp_type, perf in perfs.items()
                }
                for source, perfs in self.performance.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved ensemble state to {filepath}")

    def load_state(self, filepath: str):
        """Load model state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.weights = state.get("weights", self.weights)
            self.bias_corrections = state.get("bias_corrections", {})

            # Reconstruct performance objects
            for source, perfs in state.get("performance", {}).items():
                self.performance[source] = {}
                for temp_type, perf_data in perfs.items():
                    self.performance[source][temp_type] = ModelPerformance(
                        source=source,
                        total_forecasts=perf_data["total_forecasts"],
                        sum_error=perf_data["sum_error"],
                        sum_abs_error=perf_data["sum_abs_error"],
                        sum_squared_error=perf_data["sum_squared_error"]
                    )

            logger.info(f"Loaded ensemble state from {filepath}")

        except FileNotFoundError:
            logger.warning(f"No saved state found at {filepath}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")


# Example usage and testing
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    print("Testing EnsembleModel...")

    model = EnsembleModel()

    # Test forecast combination
    forecasts = {
        "ecmwf": {"high": 87.0, "low": 72.0},
        "gfs": {"high": 86.0, "low": 71.0},
        "nws": {"high": 85.0, "low": 70.0}
    }

    ensemble = model.combine_forecasts(
        forecasts,
        target_date=date.today() + timedelta(days=1),
        location="NYC"
    )

    print(f"\nCombined forecast:")
    print(f"  High: {ensemble.high_temp:.1f}°F (±{ensemble.high_std:.1f})")
    print(f"  Low: {ensemble.low_temp:.1f}°F (±{ensemble.low_std:.1f})")
    print(f"  Model agreement: {ensemble.model_agreement:.2f}")

    # Test probability calculation
    prob_result = model.calculate_probability(
        forecasts, threshold=85.0, days_out=1, temp_type="high"
    )
    print(f"\nP(High > 85°F):")
    print(f"  Ensemble: {prob_result['ensemble_prob']:.3f}")
    print(f"  By model: {prob_result['model_probs']}")

    # Simulate some outcomes for performance tracking
    print("\nSimulating 20 outcomes...")
    import random
    for i in range(20):
        fake_forecasts = {
            "ecmwf": {"high": 75 + random.gauss(0, 2), "low": 60 + random.gauss(0, 2)},
            "gfs": {"high": 75 + random.gauss(1, 3), "low": 60 + random.gauss(1, 3)},
            "nws": {"high": 75 + random.gauss(-1, 2.5), "low": 60 + random.gauss(-1, 2.5)}
        }
        actual_high = 75 + random.gauss(0, 2)
        actual_low = 60 + random.gauss(0, 2)
        model.record_outcome(fake_forecasts, actual_high, actual_low)

    # Get performance summary
    summary = model.get_performance_summary()
    print("\nModel performance:")
    for source, stats in summary.items():
        print(f"  {source}:")
        print(f"    High RMSE: {stats['high_rmse']:.2f}°F, Bias: {stats['high_bias']:+.2f}°F")
        print(f"    Low RMSE: {stats['low_rmse']:.2f}°F, Bias: {stats['low_bias']:+.2f}°F")

    # Update weights
    model.update_weights(min_samples=10)
    print(f"\nUpdated weights: {model.weights}")

    model.update_bias_corrections(min_samples=10)
    print(f"Bias corrections: {model.bias_corrections}")

    print("\n✓ Ensemble model tests passed!")
