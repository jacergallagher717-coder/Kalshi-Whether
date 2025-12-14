"""Tests for probability calculations."""

import pytest
from models.probability import (
    forecast_to_probability,
    ensemble_probability,
    calculate_probability_bounds,
    get_breakeven_temp
)


class TestForecastToProbability:
    """Tests for forecast_to_probability function."""

    def test_forecast_above_threshold(self):
        """When forecast > threshold, probability should be > 0.5."""
        prob = forecast_to_probability(85.0, 82.0, 1, "above")
        assert prob > 0.5
        assert prob < 1.0

    def test_forecast_below_threshold(self):
        """When forecast < threshold, probability should be < 0.5."""
        prob = forecast_to_probability(80.0, 85.0, 1, "above")
        assert prob < 0.5
        assert prob > 0.0

    def test_forecast_equals_threshold(self):
        """When forecast = threshold, probability should be ~0.5."""
        prob = forecast_to_probability(85.0, 85.0, 1, "above")
        assert 0.45 < prob < 0.55

    def test_direction_below(self):
        """Test 'below' direction calculation."""
        prob_above = forecast_to_probability(85.0, 82.0, 1, "above")
        prob_below = forecast_to_probability(85.0, 82.0, 1, "below")
        # P(above) + P(below) should approximately equal 1
        assert abs((prob_above + prob_below) - 1.0) < 0.01

    def test_uncertainty_increases_with_days(self):
        """Probability should be closer to 0.5 for further forecasts."""
        prob_1day = forecast_to_probability(87.0, 85.0, 1, "above")
        prob_5day = forecast_to_probability(87.0, 85.0, 5, "above")

        # 1-day forecast should be more confident (further from 0.5)
        assert abs(prob_1day - 0.5) > abs(prob_5day - 0.5)

    def test_probability_bounds(self):
        """Probability should always be between 0.001 and 0.999."""
        # Extreme case: forecast way above threshold
        prob = forecast_to_probability(100.0, 50.0, 1, "above")
        assert 0.001 <= prob <= 0.999

        # Extreme case: forecast way below threshold
        prob = forecast_to_probability(50.0, 100.0, 1, "above")
        assert 0.001 <= prob <= 0.999


class TestEnsembleProbability:
    """Tests for ensemble_probability function."""

    def test_basic_ensemble(self):
        """Test basic ensemble calculation."""
        forecasts = {"ecmwf": 87.0, "gfs": 86.0, "nws": 85.0}
        result = ensemble_probability(forecasts, 85.0, 1, "above")

        assert "ensemble_prob" in result
        assert "model_probs" in result
        assert "model_spread" in result
        assert "confidence" in result
        assert 0 < result["ensemble_prob"] < 1

    def test_all_models_agree(self):
        """When models agree, spread should be low and confidence high."""
        forecasts = {"ecmwf": 85.0, "gfs": 85.0, "nws": 85.0}
        result = ensemble_probability(forecasts, 85.0, 1, "above")

        assert result["model_spread"] < 1.0
        assert result["confidence"] > 0.5

    def test_models_disagree(self):
        """When models disagree, spread should be high and confidence lower."""
        forecasts = {"ecmwf": 90.0, "gfs": 80.0, "nws": 85.0}
        result = ensemble_probability(forecasts, 85.0, 1, "above")

        assert result["model_spread"] > 3.0
        assert result["confidence"] < 0.8

    def test_missing_models(self):
        """Test with missing model data."""
        forecasts = {"ecmwf": 85.0}  # Only one model
        result = ensemble_probability(forecasts, 85.0, 1, "above")

        assert result["ensemble_prob"] is not None
        assert len(result["model_probs"]) == 1

    def test_empty_forecasts(self):
        """Test with empty forecasts."""
        result = ensemble_probability({}, 85.0, 1, "above")

        assert result["ensemble_prob"] == 0.5
        assert result["confidence"] == 0.0


class TestBreakevenTemp:
    """Tests for get_breakeven_temp function."""

    def test_breakeven_roundtrip(self):
        """Verify breakeven temp gives back the same probability."""
        target_prob = 0.40
        threshold = 85.0
        days_out = 1

        implied_temp = get_breakeven_temp(threshold, days_out, target_prob, "above")
        check_prob = forecast_to_probability(implied_temp, threshold, days_out, "above")

        assert abs(check_prob - target_prob) < 0.01

    def test_higher_prob_higher_temp(self):
        """Higher probability should imply higher temperature (for 'above')."""
        temp_40 = get_breakeven_temp(85.0, 1, 0.40, "above")
        temp_60 = get_breakeven_temp(85.0, 1, 0.60, "above")

        assert temp_60 > temp_40


class TestProbabilityBounds:
    """Tests for calculate_probability_bounds function."""

    def test_bounds_contain_center(self):
        """Bounds should contain the center probability estimate."""
        low, high = calculate_probability_bounds(85.0, 82.0, 1, "above")
        center = forecast_to_probability(85.0, 82.0, 1, "above")

        assert low <= center <= high

    def test_bounds_widen_with_days(self):
        """Bounds should be wider for longer forecast horizons."""
        low_1, high_1 = calculate_probability_bounds(85.0, 82.0, 1, "above")
        low_5, high_5 = calculate_probability_bounds(85.0, 82.0, 5, "above")

        range_1 = high_1 - low_1
        range_5 = high_5 - low_5

        assert range_5 > range_1
