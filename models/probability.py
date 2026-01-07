"""
Convert temperature point forecasts into probability distributions.

Core concept: A forecast of "High: 85°F" doesn't mean exactly 85°F.
It means a distribution centered around 85°F with uncertainty.

Method:
1. Take point forecast (e.g., 85°F)
2. Apply uncertainty based on forecast horizon (see TEMP_UNCERTAINTY in settings)
3. Use normal distribution to calculate P(temp > threshold)
"""

from scipy import stats
import numpy as np
from typing import Dict, Optional, Tuple

from config.settings import TEMP_UNCERTAINTY, MODEL_WEIGHTS, CITY_MODEL_WEIGHTS, USE_CITY_WEIGHTS
from utils.logger import get_logger

logger = get_logger("probability")


def get_model_weights_for_city(city: str = None) -> Dict[str, float]:
    """
    Get model weights, optionally customized for a specific city.

    Cities have different weather patterns and some models perform better
    in certain regions. For example:
    - ECMWF tends to be best for coastal cities (NYC, LA, MIA)
    - GFS can be better for continental cities (CHI, DEN)

    Args:
        city: City code (NYC, CHI, LA, etc.) or None for defaults

    Returns:
        Dict of model_name -> weight (sums to 1.0)
    """
    if not USE_CITY_WEIGHTS or not city:
        return MODEL_WEIGHTS

    # Check if we have city-specific weights
    city_weights = CITY_MODEL_WEIGHTS.get(city)
    if city_weights:
        logger.debug(f"Using city-specific weights for {city}")
        return city_weights

    # Fall back to defaults
    return MODEL_WEIGHTS


def forecast_to_probability(
    forecast_temp: float,
    threshold: float,
    days_out: int,
    direction: str = "above"
) -> float:
    """
    Convert a temperature forecast to probability of exceeding threshold.

    Uses a normal distribution centered on the forecast with standard deviation
    that increases with forecast horizon.

    Args:
        forecast_temp: Forecasted temperature in °F
        threshold: Market strike price (e.g., 85°F)
        days_out: Days until settlement (affects uncertainty)
        direction: "above" for P(T > threshold), "below" for P(T < threshold)

    Returns:
        Probability between 0 and 1

    Example:
        >>> forecast_to_probability(85.0, 82.0, 1, "above")
        0.933  # ~93% chance temp > 82 when forecast is 85 with 2°F std dev
    """
    # Get uncertainty for this forecast horizon
    # Same-day (0) has tightest uncertainty, increases with days out
    if days_out < 0:
        std_dev = TEMP_UNCERTAINTY[0]  # Use same-day for past dates (shouldn't happen)
    elif days_out > 7:
        std_dev = TEMP_UNCERTAINTY[7]  # Cap at 7-day uncertainty
    else:
        std_dev = TEMP_UNCERTAINTY.get(days_out, TEMP_UNCERTAINTY[1])

    # Create normal distribution centered on forecast
    dist = stats.norm(loc=forecast_temp, scale=std_dev)

    if direction == "above":
        # P(T > threshold) = 1 - CDF(threshold)
        prob = 1 - dist.cdf(threshold)
    else:
        # P(T < threshold) = CDF(threshold)
        prob = dist.cdf(threshold)

    # Ensure probability is in valid range
    return max(0.001, min(0.999, prob))


def ensemble_probability(
    forecasts: Dict[str, float],
    threshold: float,
    days_out: int,
    direction: str = "above",
    weights: Dict[str, float] = None,
    city: str = None
) -> Dict[str, any]:
    """
    Combine multiple forecast sources into ensemble probability.

    Uses weighted average of individual model probabilities, with optional
    adjustments based on model spread (disagreement).

    Args:
        forecasts: Dict mapping source to forecast temp {"ecmwf": 85.0, "gfs": 84.0}
        threshold: Temperature threshold for the market
        days_out: Days until settlement
        direction: "above" or "below"
        weights: Optional custom weights. Defaults to MODEL_WEIGHTS.
        city: Optional city code for city-specific model weights.

    Returns:
        Dictionary with:
        - ensemble_prob: Weighted average probability
        - model_probs: Individual model probabilities
        - model_spread: Standard deviation of forecasts (disagreement indicator)
        - confidence: How confident we are in the ensemble (0-1)
        - ensemble_temp: Weighted average temperature forecast
    """
    # Use provided weights, or city-specific weights, or defaults
    if weights:
        effective_weights = weights
    elif city:
        effective_weights = get_model_weights_for_city(city)
    else:
        effective_weights = MODEL_WEIGHTS

    # Calculate probability from each model
    model_probs = {}
    valid_forecasts = {}

    for source, temp in forecasts.items():
        if temp is not None:
            prob = forecast_to_probability(temp, threshold, days_out, direction)
            model_probs[source] = prob
            valid_forecasts[source] = temp

    if not model_probs:
        logger.warning("No valid forecasts for ensemble calculation")
        return {
            "ensemble_prob": 0.5,
            "model_probs": {},
            "model_spread": 0.0,
            "confidence": 0.0,
            "ensemble_temp": None
        }

    # Calculate weighted average probability
    total_weight = 0.0
    weighted_prob_sum = 0.0
    weighted_temp_sum = 0.0

    for source, prob in model_probs.items():
        weight = effective_weights.get(source, 0.2)  # Default weight if not specified
        weighted_prob_sum += prob * weight
        weighted_temp_sum += valid_forecasts[source] * weight
        total_weight += weight

    ensemble_prob = weighted_prob_sum / total_weight if total_weight > 0 else 0.5
    ensemble_temp = weighted_temp_sum / total_weight if total_weight > 0 else None

    # Calculate model spread (disagreement)
    temps = list(valid_forecasts.values())
    model_spread = np.std(temps) if len(temps) > 1 else 0.0

    # Calculate confidence based on:
    # 1. Number of models agreeing
    # 2. Model spread (lower spread = higher confidence)
    # 3. Forecast horizon (shorter = higher confidence)

    num_models = len(model_probs)
    max_models = len(MODEL_WEIGHTS)

    # Base confidence from number of models (more models = more confidence)
    model_confidence = num_models / max_models

    # Adjust for spread (high spread reduces confidence)
    # If models disagree by more than 5°F, significantly reduce confidence
    spread_factor = max(0.3, 1.0 - (model_spread / 10.0))

    # Adjust for forecast horizon (further out = less confident)
    horizon_factor = max(0.5, 1.0 - (days_out - 1) * 0.1)

    confidence = model_confidence * spread_factor * horizon_factor
    confidence = max(0.1, min(1.0, confidence))

    return {
        "ensemble_prob": ensemble_prob,
        "model_probs": model_probs,
        "model_spread": model_spread,
        "confidence": confidence,
        "ensemble_temp": ensemble_temp
    }


def calculate_probability_bounds(
    forecast_temp: float,
    threshold: float,
    days_out: int,
    direction: str = "above",
    confidence_interval: float = 0.8
) -> Tuple[float, float]:
    """
    Calculate confidence interval for probability estimate.

    Accounts for uncertainty in both the forecast and the model.

    Args:
        forecast_temp: Forecasted temperature
        threshold: Temperature threshold
        days_out: Days until settlement
        direction: "above" or "below"
        confidence_interval: Width of confidence interval (0-1)

    Returns:
        Tuple of (lower_bound, upper_bound) probabilities
    """
    base_std = TEMP_UNCERTAINTY.get(days_out, 8.0)

    # Calculate base probability
    center_prob = forecast_to_probability(forecast_temp, threshold, days_out, direction)

    # Calculate probabilities at +/- 1 std dev forecast uncertainty
    # This represents uncertainty in our forecast estimate
    low_temp = forecast_temp - base_std * 0.5
    high_temp = forecast_temp + base_std * 0.5

    if direction == "above":
        # Higher temp = higher prob of exceeding threshold
        low_prob = forecast_to_probability(low_temp, threshold, days_out, direction)
        high_prob = forecast_to_probability(high_temp, threshold, days_out, direction)
    else:
        # Lower temp = higher prob of being below threshold
        low_prob = forecast_to_probability(high_temp, threshold, days_out, direction)
        high_prob = forecast_to_probability(low_temp, threshold, days_out, direction)

    return (low_prob, high_prob)


def get_breakeven_temp(
    threshold: float,
    days_out: int,
    target_prob: float,
    direction: str = "above"
) -> float:
    """
    Calculate what forecast temp would give a specific probability.

    Useful for understanding: "What does the market think the temp will be?"

    Args:
        threshold: Temperature threshold
        days_out: Days until settlement
        target_prob: Desired probability (e.g., market implied prob)
        direction: "above" or "below"

    Returns:
        Temperature that would result in target_prob
    """
    std_dev = TEMP_UNCERTAINTY.get(days_out, 8.0)

    if direction == "above":
        # P(T > threshold) = target_prob
        # 1 - CDF(threshold) = target_prob
        # CDF(threshold) = 1 - target_prob
        # threshold = mu + sigma * ppf(1 - target_prob)
        # mu = threshold - sigma * ppf(1 - target_prob)
        z_score = stats.norm.ppf(1 - target_prob)
        implied_temp = threshold - std_dev * z_score
    else:
        # P(T < threshold) = target_prob
        # CDF(threshold) = target_prob
        # threshold = mu + sigma * ppf(target_prob)
        # mu = threshold - sigma * ppf(target_prob)
        z_score = stats.norm.ppf(target_prob)
        implied_temp = threshold - std_dev * z_score

    return implied_temp


# Unit tests
if __name__ == "__main__":
    print("Testing probability calculations...")

    # Test basic probability calculation
    prob = forecast_to_probability(85.0, 82.0, 1, "above")
    print(f"\nP(T > 82°F | forecast=85°F, 1 day out) = {prob:.3f}")
    assert 0.9 < prob < 0.99, f"Expected ~0.93, got {prob}"

    prob = forecast_to_probability(85.0, 88.0, 1, "above")
    print(f"P(T > 88°F | forecast=85°F, 1 day out) = {prob:.3f}")
    assert 0.01 < prob < 0.15, f"Expected ~0.07, got {prob}"

    # Test ensemble probability
    forecasts = {"ecmwf": 87.0, "gfs": 86.0, "nws": 85.0}
    result = ensemble_probability(forecasts, 85.0, 1, "above")
    print(f"\nEnsemble for NYC High > 85°F:")
    print(f"  Forecasts: {forecasts}")
    print(f"  Ensemble probability: {result['ensemble_prob']:.3f}")
    print(f"  Model probabilities: {result['model_probs']}")
    print(f"  Model spread: {result['model_spread']:.2f}°F")
    print(f"  Confidence: {result['confidence']:.2f}")

    # Test breakeven calculation
    implied = get_breakeven_temp(85.0, 1, 0.40, "above")
    print(f"\nMarket at 40%: Implied forecast = {implied:.1f}°F")

    # Verify: if we plug implied temp back in, we should get ~40%
    check_prob = forecast_to_probability(implied, 85.0, 1, "above")
    print(f"Verification: P(T > 85 | forecast={implied:.1f}) = {check_prob:.3f}")
    assert abs(check_prob - 0.40) < 0.01, f"Breakeven calculation error"

    print("\n✓ All probability tests passed!")
