"""Tests for utility helper functions."""

import pytest
from datetime import date
from utils.helpers import (
    parse_kalshi_ticker,
    format_kalshi_ticker,
    calculate_days_until,
    price_to_probability,
    calculate_pnl,
    format_currency,
    format_percent
)


class TestParseKalshiTicker:
    """Tests for parse_kalshi_ticker function."""

    def test_parse_high_temp_ticker(self):
        """Test parsing a high temperature ticker."""
        result = parse_kalshi_ticker("HIGHNY-25JAN15-T35")

        assert result["ticker"] == "HIGHNY-25JAN15-T35"
        assert result["location"] == "NYC"
        assert result["market_type"] == "high"
        assert result["target_date"] == date(2025, 1, 15)
        assert result["temp_threshold"] == 35.0

    def test_parse_low_temp_ticker(self):
        """Test parsing a low temperature ticker."""
        result = parse_kalshi_ticker("LOWNY-25FEB20-T20")

        assert result["location"] == "NYC"
        assert result["market_type"] == "low"
        assert result["target_date"] == date(2025, 2, 20)
        assert result["temp_threshold"] == 20.0

    def test_parse_negative_temp(self):
        """Test parsing a ticker with negative temperature."""
        result = parse_kalshi_ticker("LOWNY-25JAN10-T-10")

        assert result["temp_threshold"] == -10.0

    def test_parse_chicago_ticker(self):
        """Test parsing a Chicago ticker."""
        result = parse_kalshi_ticker("HIGHCHI-25MAR01-T50")

        assert result["location"] == "CHI"
        assert result["market_type"] == "high"
        assert result["target_date"] == date(2025, 3, 1)

    def test_parse_invalid_ticker_format(self):
        """Test that invalid tickers raise ValueError."""
        with pytest.raises(ValueError):
            parse_kalshi_ticker("INVALID-TICKER")

    def test_parse_invalid_month(self):
        """Test that invalid month raises ValueError."""
        with pytest.raises(ValueError):
            parse_kalshi_ticker("HIGHNY-25XYZ15-T35")


class TestFormatKalshiTicker:
    """Tests for format_kalshi_ticker function."""

    def test_format_ticker(self):
        """Test formatting a ticker from components."""
        ticker = format_kalshi_ticker("NYC", "high", date(2025, 1, 15), 35)
        assert ticker == "HIGHNY-25JAN15-T35"

    def test_format_roundtrip(self):
        """Test that parse and format are inverse operations."""
        original = "HIGHNY-25JAN15-T35"
        parsed = parse_kalshi_ticker(original)
        formatted = format_kalshi_ticker(
            parsed["location"],
            parsed["market_type"],
            parsed["target_date"],
            int(parsed["temp_threshold"])
        )
        assert formatted == original


class TestCalculateDaysUntil:
    """Tests for calculate_days_until function."""

    def test_future_date(self):
        """Test calculation for future date."""
        from datetime import timedelta
        future = date.today() + timedelta(days=5)
        assert calculate_days_until(future) == 5

    def test_today(self):
        """Test calculation for today."""
        assert calculate_days_until(date.today()) == 0

    def test_past_date(self):
        """Test calculation for past date returns 0."""
        from datetime import timedelta
        past = date.today() - timedelta(days=5)
        assert calculate_days_until(past) == 0


class TestPriceToProbability:
    """Tests for price_to_probability function."""

    def test_valid_price(self):
        """Test conversion of valid price."""
        assert price_to_probability(0.5) == 0.5
        assert price_to_probability(0.75) == 0.75

    def test_bounds(self):
        """Test that result is bounded to 0-1."""
        assert price_to_probability(-0.1) == 0.0
        assert price_to_probability(1.5) == 1.0


class TestCalculatePnl:
    """Tests for calculate_pnl function."""

    def test_buy_yes_win(self):
        """Test P&L for winning YES trade."""
        gross, net = calculate_pnl(
            entry_price=0.40,
            exit_price=1.00,
            contracts=10,
            direction="BUY_YES",
            fees=0.17
        )

        # Gross = (1.00 - 0.40) * 10 = 6.00
        assert abs(gross - 6.00) < 0.01
        # Net = 6.00 - 0.17 = 5.83
        assert abs(net - 5.83) < 0.01

    def test_buy_yes_lose(self):
        """Test P&L for losing YES trade."""
        gross, net = calculate_pnl(
            entry_price=0.40,
            exit_price=0.00,
            contracts=10,
            direction="BUY_YES",
            fees=0.17
        )

        # Gross = (0.00 - 0.40) * 10 = -4.00
        assert abs(gross - (-4.00)) < 0.01
        # Net = -4.00 - 0.17 = -4.17
        assert abs(net - (-4.17)) < 0.01

    def test_buy_no_win(self):
        """Test P&L for winning NO trade."""
        gross, net = calculate_pnl(
            entry_price=0.60,  # YES price
            exit_price=0.00,  # YES loses
            contracts=10,
            direction="BUY_NO",
            fees=0.17
        )

        # For BUY_NO: profit when YES goes down
        # Gross = (0.60 - 0.00) * 10 = 6.00
        assert abs(gross - 6.00) < 0.01


class TestFormatCurrency:
    """Tests for format_currency function."""

    def test_positive(self):
        """Test formatting positive amount."""
        assert format_currency(42.50) == "$42.50"

    def test_negative(self):
        """Test formatting negative amount."""
        assert format_currency(-42.50) == "-$42.50"

    def test_zero(self):
        """Test formatting zero."""
        assert format_currency(0) == "$0.00"


class TestFormatPercent:
    """Tests for format_percent function."""

    def test_basic(self):
        """Test basic percentage formatting."""
        assert format_percent(0.5) == "50.0%"
        assert format_percent(0.123) == "12.3%"

    def test_small(self):
        """Test small percentage."""
        assert format_percent(0.001) == "0.1%"
