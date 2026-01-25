"""
Ladder Consistency Analyzer

Finds mispriced contracts in Kalshi's threshold/bracket ladders.

Key insight: Adjacent strikes should have monotonically consistent probabilities.
If "CPI > 3.0%" = 40% and "CPI > 3.1%" = 45%, that's inconsistent -
the probability of exceeding a HIGHER threshold can't be MORE than
exceeding a lower threshold.

This module:
1. Fetches all strikes for a given market series
2. Checks for logical inconsistencies
3. Identifies arbitrage opportunities
4. Calculates implied distributions
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats

from utils.logger import get_logger

logger = get_logger("ladder_analyzer")


@dataclass
class Strike:
    """Represents a single strike in a ladder."""
    ticker: str
    threshold: float  # e.g., 3.0 for "above 3.0%"
    direction: str    # "above" or "below"
    yes_bid: float    # Best bid for YES
    yes_ask: float    # Best ask for YES
    yes_mid: float    # Midpoint
    volume: int       # Trading volume
    open_interest: int

    @property
    def implied_prob(self) -> float:
        """Implied probability from mid price."""
        return self.yes_mid

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.yes_ask - self.yes_bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        if self.yes_mid <= 0:
            return float('inf')
        return self.spread / self.yes_mid


@dataclass
class Inconsistency:
    """Represents a detected pricing inconsistency."""
    type: str  # "monotonicity", "sum_exceeds_100", "negative_density"
    strikes: List[Strike]
    description: str
    arbitrage_value: float  # Potential profit from exploiting
    confidence: float  # How confident we are this is real mispricing


class LadderAnalyzer:
    """
    Analyzes price ladders for inconsistencies and arbitrage opportunities.

    Usage:
        analyzer = LadderAnalyzer()
        strikes = [Strike(...), Strike(...), ...]
        issues = analyzer.find_inconsistencies(strikes)
        for issue in issues:
            print(f"Found: {issue.description}, value: ${issue.arbitrage_value}")
    """

    def __init__(self, min_spread_pct: float = 0.05):
        """
        Args:
            min_spread_pct: Minimum spread percentage to consider a market liquid
        """
        self.min_spread_pct = min_spread_pct

    def find_inconsistencies(self, strikes: List[Strike]) -> List[Inconsistency]:
        """
        Find all pricing inconsistencies in a ladder.

        Args:
            strikes: List of Strike objects for a market series

        Returns:
            List of Inconsistency objects describing issues found
        """
        issues = []

        # Sort by threshold
        sorted_strikes = sorted(strikes, key=lambda s: s.threshold)

        # Check 1: Monotonicity for "above" markets
        above_strikes = [s for s in sorted_strikes if s.direction == "above"]
        if len(above_strikes) >= 2:
            mono_issues = self._check_monotonicity_above(above_strikes)
            issues.extend(mono_issues)

        # Check 2: Monotonicity for "below" markets
        below_strikes = [s for s in sorted_strikes if s.direction == "below"]
        if len(below_strikes) >= 2:
            mono_issues = self._check_monotonicity_below(below_strikes)
            issues.extend(mono_issues)

        # Check 3: Adjacent strike probabilities imply positive density
        if len(sorted_strikes) >= 2:
            density_issues = self._check_positive_density(sorted_strikes)
            issues.extend(density_issues)

        # Check 4: Implied distribution reasonableness
        dist_issues = self._check_distribution_reasonableness(sorted_strikes)
        issues.extend(dist_issues)

        return issues

    def _check_monotonicity_above(self, strikes: List[Strike]) -> List[Inconsistency]:
        """
        For "above X" markets, higher thresholds should have lower probability.

        P(CPI > 3.0%) >= P(CPI > 3.1%) >= P(CPI > 3.2%) ...

        If this is violated, there's an arbitrage.
        """
        issues = []
        sorted_strikes = sorted(strikes, key=lambda s: s.threshold)

        for i in range(len(sorted_strikes) - 1):
            lower = sorted_strikes[i]
            higher = sorted_strikes[i + 1]

            # Higher threshold should have LOWER probability
            if higher.yes_mid > lower.yes_mid:
                # This is wrong - arbitrage opportunity
                arb_value = self._calculate_arb_value(lower, higher, "above_mono")

                issues.append(Inconsistency(
                    type="monotonicity",
                    strikes=[lower, higher],
                    description=(
                        f"Monotonicity violation: P(>{higher.threshold}) = {higher.yes_mid:.1%} > "
                        f"P(>{lower.threshold}) = {lower.yes_mid:.1%}. "
                        f"Should be reversed."
                    ),
                    arbitrage_value=arb_value,
                    confidence=self._calculate_confidence(lower, higher)
                ))

        return issues

    def _check_monotonicity_below(self, strikes: List[Strike]) -> List[Inconsistency]:
        """
        For "below X" markets, higher thresholds should have higher probability.

        P(CPI < 3.0%) <= P(CPI < 3.1%) <= P(CPI < 3.2%) ...
        """
        issues = []
        sorted_strikes = sorted(strikes, key=lambda s: s.threshold)

        for i in range(len(sorted_strikes) - 1):
            lower = sorted_strikes[i]
            higher = sorted_strikes[i + 1]

            # Higher threshold should have HIGHER probability for "below"
            if higher.yes_mid < lower.yes_mid:
                arb_value = self._calculate_arb_value(lower, higher, "below_mono")

                issues.append(Inconsistency(
                    type="monotonicity",
                    strikes=[lower, higher],
                    description=(
                        f"Monotonicity violation: P(<{higher.threshold}) = {higher.yes_mid:.1%} < "
                        f"P(<{lower.threshold}) = {lower.yes_mid:.1%}. "
                        f"Should be reversed."
                    ),
                    arbitrage_value=arb_value,
                    confidence=self._calculate_confidence(lower, higher)
                ))

        return issues

    def _check_positive_density(self, strikes: List[Strike]) -> List[Inconsistency]:
        """
        Check that implied probability density is positive between adjacent strikes.

        For "above" markets:
        P(threshold1 < X < threshold2) = P(X > threshold1) - P(X > threshold2)

        This should always be >= 0.
        """
        issues = []

        above_strikes = sorted(
            [s for s in strikes if s.direction == "above"],
            key=lambda s: s.threshold
        )

        for i in range(len(above_strikes) - 1):
            lower = above_strikes[i]
            higher = above_strikes[i + 1]

            # Implied probability in the range (lower, higher]
            range_prob = lower.yes_mid - higher.yes_mid

            if range_prob < -0.01:  # Allow 1% tolerance for bid-ask
                issues.append(Inconsistency(
                    type="negative_density",
                    strikes=[lower, higher],
                    description=(
                        f"Negative implied density between {lower.threshold} and {higher.threshold}: "
                        f"{range_prob:.1%}. Markets imply impossible distribution."
                    ),
                    arbitrage_value=abs(range_prob) * 10,  # Scale for position sizing
                    confidence=0.9 if abs(range_prob) > 0.03 else 0.6
                ))

        return issues

    def _check_distribution_reasonableness(self, strikes: List[Strike]) -> List[Inconsistency]:
        """
        Check if implied distribution is reasonable (not too far from normal).

        Extreme skew or kurtosis might indicate mispricing.
        """
        issues = []

        above_strikes = sorted(
            [s for s in strikes if s.direction == "above"],
            key=lambda s: s.threshold
        )

        if len(above_strikes) < 3:
            return issues

        # Extract implied CDF
        thresholds = [s.threshold for s in above_strikes]
        probs = [s.yes_mid for s in above_strikes]  # P(X > threshold)

        # Convert to P(X <= threshold) = 1 - P(X > threshold)
        cdf_values = [1 - p for p in probs]

        # Try to fit a normal distribution
        try:
            # Use midpoint of range as rough estimate of mean
            implied_mean = sum(t * (cdf_values[i] - (cdf_values[i-1] if i > 0 else 0))
                              for i, t in enumerate(thresholds))

            # Check if any strike has extreme deviation from "reasonable"
            for strike in above_strikes:
                # Compare to what a normal dist would imply
                # This is a simplified check
                if strike.yes_mid > 0.95 and strike.threshold > implied_mean + 1:
                    issues.append(Inconsistency(
                        type="extreme_probability",
                        strikes=[strike],
                        description=(
                            f"Extreme probability {strike.yes_mid:.1%} for threshold {strike.threshold} "
                            f"seems inconsistent with implied mean ~{implied_mean:.2f}"
                        ),
                        arbitrage_value=0.05,
                        confidence=0.5  # Lower confidence, could be legitimate
                    ))

        except Exception as e:
            logger.debug(f"Distribution check failed: {e}")

        return issues

    def _calculate_arb_value(self, strike1: Strike, strike2: Strike, arb_type: str) -> float:
        """
        Calculate potential arbitrage value.

        For monotonicity violations, the arb is:
        - Buy the underpriced strike
        - Sell the overpriced strike
        - Guaranteed profit = |price difference| - transaction costs
        """
        price_diff = abs(strike1.yes_mid - strike2.yes_mid)

        # Account for bid-ask spread (you'll buy at ask, sell at bid)
        total_spread_cost = strike1.spread + strike2.spread

        # Net arbitrage value
        arb_value = price_diff - total_spread_cost

        # Only positive if arb exceeds spread costs
        return max(0, arb_value)

    def _calculate_confidence(self, strike1: Strike, strike2: Strike) -> float:
        """
        Calculate confidence that this is a real mispricing vs noise.

        Higher confidence when:
        - Both strikes are liquid (narrow spreads)
        - Price difference is large
        - Volume is meaningful
        """
        # Liquidity factor
        avg_spread = (strike1.spread_pct + strike2.spread_pct) / 2
        liquidity_score = max(0, 1 - avg_spread * 10)  # Penalize wide spreads

        # Magnitude factor
        price_diff = abs(strike1.yes_mid - strike2.yes_mid)
        magnitude_score = min(1, price_diff / 0.10)  # 10% diff = full score

        # Volume factor
        total_volume = (strike1.volume or 0) + (strike2.volume or 0)
        volume_score = min(1, total_volume / 1000)  # 1000 volume = full score

        # Weighted average
        confidence = (
            0.4 * liquidity_score +
            0.4 * magnitude_score +
            0.2 * volume_score
        )

        return round(confidence, 2)

    def get_implied_distribution(self, strikes: List[Strike]) -> Optional[Dict]:
        """
        Calculate the implied probability distribution from strike prices.

        Returns dict with:
        - mean: Implied expected value
        - std: Implied standard deviation
        - skew: Implied skewness
        - pdf: Discrete PDF values
        """
        above_strikes = sorted(
            [s for s in strikes if s.direction == "above"],
            key=lambda s: s.threshold
        )

        if len(above_strikes) < 2:
            return None

        thresholds = [s.threshold for s in above_strikes]
        survival_probs = [s.yes_mid for s in above_strikes]  # P(X > t)

        # Calculate implied PDF (probability mass in each bucket)
        pdf = []
        for i in range(len(thresholds) - 1):
            # P(t_i < X <= t_{i+1}) = P(X > t_i) - P(X > t_{i+1})
            prob_mass = survival_probs[i] - survival_probs[i + 1]
            midpoint = (thresholds[i] + thresholds[i + 1]) / 2
            pdf.append({
                'lower': thresholds[i],
                'upper': thresholds[i + 1],
                'midpoint': midpoint,
                'probability': max(0, prob_mass)
            })

        # Calculate moments
        total_prob = sum(b['probability'] for b in pdf)
        if total_prob <= 0:
            return None

        # Normalize
        for b in pdf:
            b['probability'] /= total_prob

        # Mean
        mean = sum(b['midpoint'] * b['probability'] for b in pdf)

        # Variance
        variance = sum(b['probability'] * (b['midpoint'] - mean)**2 for b in pdf)
        std = variance ** 0.5 if variance > 0 else 0

        # Skewness (simplified)
        if std > 0:
            skew = sum(b['probability'] * ((b['midpoint'] - mean) / std)**3 for b in pdf)
        else:
            skew = 0

        return {
            'mean': round(mean, 3),
            'std': round(std, 3),
            'skew': round(skew, 3),
            'pdf': pdf,
            'thresholds': thresholds,
            'survival_probs': survival_probs
        }


def analyze_cpi_ladder(kalshi_client, market_series: str = "KXCPIYOY") -> Dict:
    """
    Convenience function to analyze CPI market ladder.

    Args:
        kalshi_client: Initialized Kalshi API client
        market_series: The series ticker (e.g., "KXCPIYOY" for CPI YoY)

    Returns:
        Analysis results including inconsistencies and implied distribution
    """
    # This would fetch from Kalshi API
    # For now, return placeholder
    logger.info(f"Analyzing {market_series} ladder...")

    analyzer = LadderAnalyzer()

    # In production, fetch strikes from Kalshi API
    # strikes = kalshi_client.get_market_series(market_series)
    # issues = analyzer.find_inconsistencies(strikes)

    return {
        'market_series': market_series,
        'inconsistencies': [],
        'implied_distribution': None,
        'recommendation': 'Implement API integration to analyze live data'
    }


if __name__ == "__main__":
    print("Testing Ladder Analyzer...")

    # Create some test strikes (simulated CPI "above" market)
    test_strikes = [
        Strike("CPI-ABOVE-2.5", 2.5, "above", 0.88, 0.92, 0.90, 500, 1000),
        Strike("CPI-ABOVE-2.6", 2.6, "above", 0.75, 0.80, 0.775, 400, 800),
        Strike("CPI-ABOVE-2.7", 2.7, "above", 0.55, 0.60, 0.575, 600, 1200),
        Strike("CPI-ABOVE-2.8", 2.8, "above", 0.30, 0.35, 0.325, 300, 600),
        Strike("CPI-ABOVE-2.9", 2.9, "above", 0.12, 0.18, 0.15, 200, 400),
    ]

    analyzer = LadderAnalyzer()

    print("\nChecking for inconsistencies...")
    issues = analyzer.find_inconsistencies(test_strikes)
    if issues:
        for issue in issues:
            print(f"  ISSUE: {issue.description}")
            print(f"    Arb value: ${issue.arbitrage_value:.2f}")
            print(f"    Confidence: {issue.confidence:.0%}")
    else:
        print("  No inconsistencies found (ladder is properly priced)")

    print("\nImplied distribution:")
    dist = analyzer.get_implied_distribution(test_strikes)
    if dist:
        print(f"  Implied Mean CPI: {dist['mean']:.2f}%")
        print(f"  Implied Std Dev: {dist['std']:.2f}%")
        print(f"  Implied Skew: {dist['skew']:.2f}")

    # Test with an inconsistent ladder
    print("\n\nTesting with inconsistent ladder...")
    bad_strikes = [
        Strike("CPI-ABOVE-2.5", 2.5, "above", 0.50, 0.55, 0.525, 500, 1000),
        Strike("CPI-ABOVE-2.6", 2.6, "above", 0.60, 0.65, 0.625, 400, 800),  # WRONG: higher than 2.5!
        Strike("CPI-ABOVE-2.7", 2.7, "above", 0.40, 0.45, 0.425, 600, 1200),
    ]

    issues = analyzer.find_inconsistencies(bad_strikes)
    for issue in issues:
        print(f"  FOUND: {issue.description}")
        print(f"    Arb value: ${issue.arbitrage_value:.3f}")
