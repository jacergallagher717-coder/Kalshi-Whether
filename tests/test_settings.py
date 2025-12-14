"""Tests for configuration settings."""

import pytest
from config.settings import calculate_kalshi_fee, get_confidence_level


class TestCalculateKalshiFee:
    """Tests for calculate_kalshi_fee function."""

    def test_fee_at_50_cents(self):
        """Fee should be maximum at 50 cents."""
        fee = calculate_kalshi_fee(0.50, 1)
        # Fee = 0.07 * 0.50 * 0.50 = 0.0175
        assert abs(fee - 0.0175) < 0.0001

    def test_fee_cap(self):
        """Fee should be capped at $0.0175 per contract."""
        fee = calculate_kalshi_fee(0.50, 1)
        assert fee <= 0.0175

    def test_fee_at_extremes(self):
        """Fee should be lower at price extremes."""
        fee_50 = calculate_kalshi_fee(0.50, 1)
        fee_10 = calculate_kalshi_fee(0.10, 1)
        fee_90 = calculate_kalshi_fee(0.90, 1)

        assert fee_50 > fee_10
        assert fee_50 > fee_90

    def test_fee_scales_with_contracts(self):
        """Fee should scale linearly with contract count."""
        fee_1 = calculate_kalshi_fee(0.50, 1)
        fee_10 = calculate_kalshi_fee(0.50, 10)

        assert abs(fee_10 - fee_1 * 10) < 0.0001

    def test_fee_zero_contracts(self):
        """Fee should be zero for zero contracts."""
        fee = calculate_kalshi_fee(0.50, 0)
        assert fee == 0


class TestGetConfidenceLevel:
    """Tests for get_confidence_level function."""

    def test_high_confidence(self):
        """Edge >= 15% should be high confidence."""
        assert get_confidence_level(0.15) == "high"
        assert get_confidence_level(0.20) == "high"
        assert get_confidence_level(0.30) == "high"

    def test_medium_confidence(self):
        """Edge between 10-15% should be medium confidence."""
        assert get_confidence_level(0.10) == "medium"
        assert get_confidence_level(0.12) == "medium"
        assert get_confidence_level(0.14) == "medium"

    def test_low_confidence(self):
        """Edge < 10% should be low confidence."""
        assert get_confidence_level(0.05) == "low"
        assert get_confidence_level(0.09) == "low"
        assert get_confidence_level(0.00) == "low"
