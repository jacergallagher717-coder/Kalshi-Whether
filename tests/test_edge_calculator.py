"""Tests for edge calculation."""

import pytest
from datetime import date, timedelta
from models.edge_calculator import EdgeCalculator, TradeSignal


class TestEdgeCalculator:
    """Tests for EdgeCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create EdgeCalculator instance."""
        return EdgeCalculator()

    def test_calculate_edge_positive(self, calculator):
        """Test positive edge calculation."""
        edge = calculator.calculate_edge(market_price=0.40, our_probability=0.70)
        assert edge == pytest.approx(0.30)

    def test_calculate_edge_negative(self, calculator):
        """Test negative edge calculation."""
        edge = calculator.calculate_edge(market_price=0.70, our_probability=0.40)
        assert edge == pytest.approx(-0.30)

    def test_calculate_edge_zero(self, calculator):
        """Test zero edge when market price equals our probability."""
        edge = calculator.calculate_edge(market_price=0.50, our_probability=0.50)
        assert edge == 0.0

    def test_expected_value_buy_yes_positive(self, calculator):
        """Test positive EV for BUY_YES when we have edge."""
        ev = calculator.calculate_expected_value(
            our_probability=0.70,
            entry_price=0.40,
            direction="BUY_YES",
            include_fees=False
        )
        # EV = 0.70 * 0.60 - 0.30 * 0.40 = 0.42 - 0.12 = 0.30
        assert abs(ev - 0.30) < 0.01

    def test_expected_value_with_fees(self, calculator):
        """Test that fees reduce EV."""
        ev_no_fees = calculator.calculate_expected_value(
            our_probability=0.70,
            entry_price=0.40,
            direction="BUY_YES",
            include_fees=False
        )
        ev_with_fees = calculator.calculate_expected_value(
            our_probability=0.70,
            entry_price=0.40,
            direction="BUY_YES",
            include_fees=True
        )
        assert ev_with_fees < ev_no_fees

    def test_determine_direction_buy_yes(self, calculator):
        """Test direction when we think market underprices YES."""
        direction = calculator.determine_direction(
            our_probability=0.70,
            market_price=0.40
        )
        assert direction == "BUY_YES"

    def test_determine_direction_buy_no(self, calculator):
        """Test direction when we think market overprices YES."""
        direction = calculator.determine_direction(
            our_probability=0.30,
            market_price=0.60
        )
        assert direction == "BUY_NO"

    def test_kelly_fraction_bounds(self, calculator):
        """Kelly fraction should be capped at 0.25."""
        kelly = calculator.calculate_kelly_fraction(
            our_probability=0.90,  # Very high probability
            entry_price=0.40,
            direction="BUY_YES"
        )
        assert 0 <= kelly <= 0.25

    def test_kelly_fraction_zero_for_bad_edge(self, calculator):
        """Kelly should be 0 or negative when EV is negative."""
        kelly = calculator.calculate_kelly_fraction(
            our_probability=0.30,  # Below market price
            entry_price=0.40,
            direction="BUY_YES"
        )
        assert kelly <= 0

    def test_generate_signal_with_edge(self, calculator):
        """Test signal generation when edge exceeds threshold."""
        signal = calculator.generate_signal(
            ticker="HIGHNY-25JAN15-T85",
            location="NYC",
            target_date=date.today() + timedelta(days=1),
            temp_threshold=85.0,
            market_type="high",
            market_price=0.40,
            forecasts={"ecmwf": 87.0, "gfs": 86.0, "nws": 85.0}
        )

        assert signal is not None
        assert signal.edge > 0.10  # Should exceed threshold
        assert signal.direction == "BUY_YES"
        assert signal.confidence in ["high", "medium", "low"]
        assert len(signal.reasoning) > 0

    def test_generate_signal_no_edge(self, calculator):
        """Test that no signal is generated when edge is below threshold."""
        signal = calculator.generate_signal(
            ticker="HIGHNY-25JAN15-T85",
            location="NYC",
            target_date=date.today() + timedelta(days=1),
            temp_threshold=85.0,
            market_type="high",
            market_price=0.68,  # Close to our estimate
            forecasts={"ecmwf": 87.0, "gfs": 86.0, "nws": 85.0}
        )

        # Edge should be below threshold, so no signal
        assert signal is None

    def test_generate_signal_expired_market(self, calculator):
        """Test that expired markets don't generate signals."""
        signal = calculator.generate_signal(
            ticker="HIGHNY-24DEC01-T85",
            location="NYC",
            target_date=date(2024, 12, 1),  # Past date
            temp_threshold=85.0,
            market_type="high",
            market_price=0.40,
            forecasts={"ecmwf": 87.0}
        )

        assert signal is None

    def test_position_size_calculation(self, calculator):
        """Test position size calculation."""
        contracts = calculator.calculate_position_size(
            edge=0.20,
            entry_price=0.40,
            direction="BUY_YES",
            kelly_fraction=0.15,
            max_position=100
        )

        assert contracts >= 1
        assert contracts <= 50  # MAX_CONTRACTS_PER_TRADE


class TestTradeSignal:
    """Tests for TradeSignal dataclass."""

    def test_trade_signal_str(self):
        """Test string representation of TradeSignal."""
        signal = TradeSignal(
            signal_id="abc123",
            timestamp=date.today(),
            ticker="HIGHNY-25JAN15-T85",
            location="NYC",
            target_date=date.today() + timedelta(days=1),
            temp_threshold=85.0,
            market_type="high",
            market_price=0.40,
            market_implied_prob=0.40,
            our_probability=0.70,
            edge=0.30,
            expected_value=0.25,
            confidence="high",
            direction="BUY_YES",
            recommended_contracts=10,
            total_cost=4.00,
            potential_profit=5.60,
            potential_loss=4.00,
            kelly_fraction=0.15
        )

        str_repr = str(signal)
        assert "HIGHNY" in str_repr
        assert "BUY_YES" in str_repr
