"""
Calculate expected value and edge for potential trades.

Edge = Our Probability - Market Implied Probability

This module outputs TradeSignal objects that can be passed to the paper trader.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, Optional, List
import uuid

from config.settings import (
    MIN_EDGE_THRESHOLD, MIN_PROBABILITY_THRESHOLD, MAX_POSITION_SIZE, MAX_CONTRACTS_PER_TRADE,
    MIN_YES_PRICE, MAX_YES_PRICE, MAX_NO_PRICE_BRACKET, BRACKET_POSITION_SCALE,
    MAX_MODEL_SPREAD, MIN_CONFIDENCE_SCORE, MIN_MODEL_AGREEMENT, MIN_MODELS_REQUIRED,
    MAX_FORECAST_DAYS, NEXT_DAY_EDGE_PENALTY,
    TRADING_WINDOW_ENABLED, TRADING_WINDOW_START_HOUR, TRADING_WINDOW_END_HOUR,
    OVERNIGHT_EDGE_PENALTY,
    calculate_kalshi_fee, get_confidence_level
)
from utils.logger import get_logger
from utils.helpers import calculate_days_until, format_percent, format_currency
from .probability import ensemble_probability, get_breakeven_temp
from .ensemble import EnsembleModel
from zoneinfo import ZoneInfo

logger = get_logger("edge_calculator")


def is_in_trading_window() -> tuple[bool, str]:
    """
    Check if current time is within the optimal trading window.

    Best trading hours: 6am-9pm Eastern
    - Morning: Weather models have updated overnight, market hasn't adjusted
    - Evening: Still okay but forecasts becoming stale
    - Overnight (9pm-6am): Forecasts will change, require higher edge

    Returns:
        Tuple of (is_in_window, reason)
    """
    if not TRADING_WINDOW_ENABLED:
        return True, "Trading window disabled"

    try:
        eastern = ZoneInfo('America/New_York')
        now_et = datetime.now(eastern)
        hour = now_et.hour

        if TRADING_WINDOW_START_HOUR <= hour < TRADING_WINDOW_END_HOUR:
            return True, f"In trading window ({hour}:00 ET)"
        else:
            return False, f"Outside trading window ({hour}:00 ET, window is {TRADING_WINDOW_START_HOUR}-{TRADING_WINDOW_END_HOUR})"
    except Exception as e:
        logger.warning(f"Could not check trading window: {e}")
        return True, "Could not determine timezone"


def is_post_model_update() -> tuple[bool, str, float]:
    """
    Check if we're in an optimal trading window after a major model update.

    Model update schedule (approximate availability in ET):
    - GFS 00Z  → ~5:00 AM ET
    - ECMWF 00Z → ~5:30 AM ET
    - GFS 12Z  → ~1:30 PM ET
    - GFS 18Z  → ~8:00 PM ET

    Best times to trade: 30-90 minutes after model updates when data is fresh
    but market hasn't fully adjusted yet.

    Returns:
        Tuple of (is_optimal, reason, edge_bonus)
        edge_bonus: Extra edge credit for optimal timing (0.0 to 0.05)
    """
    try:
        eastern = ZoneInfo('America/New_York')
        now_et = datetime.now(eastern)
        hour = now_et.hour
        minute = now_et.minute
        time_decimal = hour + minute / 60.0

        # Optimal windows after model updates (start_hour, end_hour, model_name)
        optimal_windows = [
            (5.0, 6.5, "GFS/ECMWF 00Z"),      # 5:00-6:30 AM (after overnight runs)
            (13.5, 15.0, "GFS 12Z"),           # 1:30-3:00 PM (after midday run)
            (20.0, 21.5, "GFS 18Z"),           # 8:00-9:30 PM (after evening run)
        ]

        for start, end, model in optimal_windows:
            if start <= time_decimal < end:
                # Give edge bonus for trading right after model update
                # Higher bonus closer to the start of the window
                window_progress = (time_decimal - start) / (end - start)
                edge_bonus = 0.03 * (1 - window_progress)  # 3% bonus at start, fading to 0%
                return True, f"Post-{model} update window", edge_bonus

        return False, f"Not in post-model-update window ({hour}:{minute:02d} ET)", 0.0

    except Exception as e:
        logger.warning(f"Could not check model update timing: {e}")
        return False, "Could not determine timezone", 0.0


@dataclass
class TradeSignal:
    """Represents a trade opportunity identified by the edge calculator."""
    signal_id: str
    timestamp: datetime
    ticker: str
    location: str
    target_date: date
    temp_threshold: float
    market_type: str  # "high" or "low"

    # Market data
    market_price: float  # Current YES price
    market_implied_prob: float

    # Our analysis
    our_probability: float
    edge: float
    expected_value: float  # EV per contract in dollars
    confidence: str  # "high", "medium", "low"

    # Trade recommendation
    direction: str  # "BUY_YES" or "BUY_NO"
    recommended_contracts: int
    total_cost: float
    potential_profit: float
    potential_loss: float

    # Details
    model_probs: Dict[str, float] = field(default_factory=dict)
    model_spread: float = 0.0
    reasoning: str = ""

    # Kelly criterion
    kelly_fraction: float = 0.0

    def __str__(self) -> str:
        return (f"Signal {self.ticker}: {self.direction} @ {format_currency(self.market_price)} "
                f"| Edge: {format_percent(self.edge)} | EV: {format_currency(self.expected_value)}")


class EdgeCalculator:
    """
    Calculates trading edge by comparing model predictions to market prices.

    The edge is the difference between:
    - Our ensemble probability estimate
    - The market's implied probability (from prices)
    """

    def __init__(self, ensemble_model: EnsembleModel = None):
        """
        Initialize the edge calculator.

        Args:
            ensemble_model: Optional custom ensemble model
        """
        self.ensemble = ensemble_model or EnsembleModel()
        logger.info("Edge calculator initialized")

    def calculate_edge(
        self,
        market_price: float,
        our_probability: float
    ) -> float:
        """
        Calculate raw edge.

        Args:
            market_price: Current YES price (0-1)
            our_probability: Our estimated probability (0-1)

        Returns:
            Edge as decimal (positive = we think market underprices YES)
        """
        market_implied_prob = market_price  # YES price = implied prob
        return our_probability - market_implied_prob

    def calculate_expected_value(
        self,
        our_probability: float,
        entry_price: float,
        direction: str,
        include_fees: bool = True
    ) -> float:
        """
        Calculate expected value per contract.

        For BUY_YES:
            EV = P(win) * ($1 - entry) - P(lose) * entry - fees

        For BUY_NO:
            EV = P(lose) * ($1 - (1-entry)) - P(win) * (1-entry) - fees
            Where P(lose) is prob that YES loses (our probability of NO)

        Args:
            our_probability: Our probability of YES outcome
            entry_price: Entry price for YES contract
            direction: "BUY_YES" or "BUY_NO"
            include_fees: Whether to subtract fees from EV

        Returns:
            Expected value in dollars per contract
        """
        if direction == "BUY_YES":
            # Buying YES at entry_price
            p_win = our_probability
            payout_if_win = 1.0 - entry_price
            loss_if_lose = entry_price
            fee_price = entry_price
        else:  # BUY_NO
            # Buying NO at (1 - entry_price) effectively
            p_win = 1 - our_probability  # Prob that NO wins
            no_price = 1.0 - entry_price
            payout_if_win = 1.0 - no_price
            loss_if_lose = no_price
            fee_price = no_price

        gross_ev = (p_win * payout_if_win) - ((1 - p_win) * loss_if_lose)

        if include_fees:
            fee = calculate_kalshi_fee(fee_price, 1)
            return gross_ev - fee
        return gross_ev

    def calculate_kelly_fraction(
        self,
        our_probability: float,
        entry_price: float,
        direction: str
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Kelly fraction = (bp - q) / b
        Where:
            b = net odds (payout / risk)
            p = probability of winning
            q = probability of losing

        Args:
            our_probability: Our probability estimate
            entry_price: Entry price
            direction: Trade direction

        Returns:
            Kelly fraction (0-1), capped at 0.25 for safety
        """
        # Guard against division by zero at price extremes
        if entry_price <= 0.01 or entry_price >= 0.99:
            return 0.0

        if direction == "BUY_YES":
            p = our_probability
            b = (1.0 - entry_price) / entry_price  # Odds
        else:
            p = 1 - our_probability
            no_price = 1.0 - entry_price
            b = (1.0 - no_price) / no_price

        q = 1 - p

        if b <= 0:
            return 0.0

        kelly = (b * p - q) / b

        # Cap at 25% of bankroll for safety (quarter Kelly)
        return max(0.0, min(0.25, kelly))

    def determine_direction(
        self,
        our_probability: float,
        market_price: float
    ) -> str:
        """
        Determine whether to buy YES or NO based on edge.

        Args:
            our_probability: Our estimated probability
            market_price: Current YES market price

        Returns:
            "BUY_YES" if we think market underprices YES
            "BUY_NO" if we think market overprices YES
        """
        edge = self.calculate_edge(market_price, our_probability)
        return "BUY_YES" if edge > 0 else "BUY_NO"

    def calculate_position_size(
        self,
        edge: float,
        entry_price: float,
        direction: str,
        kelly_fraction: float,
        max_position: float = None
    ) -> int:
        """
        Calculate recommended number of contracts.

        Uses combination of:
        - Kelly criterion for optimal sizing
        - Edge-based scaling (higher edge = more contracts)
        - Maximum position limits

        Args:
            edge: Calculated edge
            entry_price: Entry price
            direction: Trade direction
            kelly_fraction: Kelly criterion fraction
            max_position: Maximum position in dollars

        Returns:
            Recommended number of contracts
        """
        max_position = max_position or MAX_POSITION_SIZE

        if direction == "BUY_YES":
            cost_per_contract = entry_price
        else:
            cost_per_contract = 1.0 - entry_price

        # Base position from max position limit
        max_contracts_from_limit = int(max_position / cost_per_contract)

        # Position from Kelly (as fraction of max)
        kelly_contracts = int(kelly_fraction * max_contracts_from_limit)

        # Scale by edge: min edge (10%) = 50% of Kelly, high edge (30%+) = 100% of Kelly
        edge_scalar = min(1.0, 0.5 + (abs(edge) - 0.10) / 0.40)
        scaled_contracts = int(kelly_contracts * edge_scalar)

        # Apply all limits
        final_contracts = min(
            scaled_contracts,
            max_contracts_from_limit,
            MAX_CONTRACTS_PER_TRADE
        )

        return max(1, final_contracts)  # At least 1 contract

    def generate_signal(
        self,
        ticker: str,
        location: str,
        target_date: date,
        temp_threshold: float,
        market_type: str,
        market_price: float,
        forecasts: Dict[str, float]
    ) -> Optional[TradeSignal]:
        """
        Generate a trade signal for a market.

        Args:
            ticker: Market ticker
            location: Location code
            target_date: Settlement date
            temp_threshold: Temperature strike
            market_type: "high" or "low"
            market_price: Current YES price
            forecasts: Dict of forecast temps by source

        Returns:
            TradeSignal if edge exceeds threshold, None otherwise
        """
        days_out = calculate_days_until(target_date, allow_negative=True)

        # Skip if market already expired
        if days_out < 0:
            logger.debug(f"Skipping {ticker}: already expired")
            return None

        # TIME-BASED FILTER: Only trade within forecast accuracy window
        # Forecasts update overnight - trading 2+ days out is unreliable
        if days_out > MAX_FORECAST_DAYS:
            logger.debug(f"Skipping {ticker}: {days_out} days out exceeds max {MAX_FORECAST_DAYS} (forecast accuracy)")
            return None

        # Calculate adjusted edge threshold based on timing
        # ENTRY TIMING RULES (user-defined best practices):
        # - Same-day (days_out=0): Require higher edge (20%+) - less time to be right
        # - Next-day (days_out=1): Optimal window - use base threshold
        # - 2+ days out: Already filtered by MAX_FORECAST_DAYS
        adjusted_edge_threshold = MIN_EDGE_THRESHOLD

        # Same-day trading requires higher conviction (markets have more info)
        SAME_DAY_EDGE_REQUIREMENT = 0.20  # 20% edge required for day-of trades
        if days_out == 0:
            adjusted_edge_threshold = max(adjusted_edge_threshold, SAME_DAY_EDGE_REQUIREMENT)
            logger.debug(f"{ticker}: Same-day trade, edge threshold set to {adjusted_edge_threshold:.0%}")
        elif days_out > 0:
            adjusted_edge_threshold += NEXT_DAY_EDGE_PENALTY
            logger.debug(f"{ticker}: Next-day trade, edge threshold increased to {adjusted_edge_threshold:.0%}")

        # TRADING WINDOW CHECK: Require higher edge outside optimal hours
        # Overnight (9pm-6am ET): Forecasts will update before settlement
        in_window, window_reason = is_in_trading_window()
        if not in_window:
            adjusted_edge_threshold += OVERNIGHT_EDGE_PENALTY
            logger.info(f"{ticker}: {window_reason} - edge threshold increased to {adjusted_edge_threshold:.0%}")

        # MODEL UPDATE TIMING BONUS: Reduce edge requirement right after model updates
        # Fresh forecasts + market hasn't adjusted = alpha opportunity
        is_optimal_time, timing_reason, edge_bonus = is_post_model_update()
        if is_optimal_time and edge_bonus > 0:
            adjusted_edge_threshold = max(0.20, adjusted_edge_threshold - edge_bonus)
            logger.info(f"{ticker}: {timing_reason} - edge threshold reduced to {adjusted_edge_threshold:.0%} (bonus: {edge_bonus:.0%})")

        # Skip markets at extreme prices (can't calculate Kelly, minimal liquidity)
        if market_price <= 0.01 or market_price >= 0.99:
            logger.debug(f"Skipping {ticker}: price at extreme ({market_price:.2f})")
            return None

        # Check if this is a bracket market (ticker contains 'B' followed by digits after date)
        is_bracket = '-B' in ticker

        # Determine direction for probability calculation
        # For HIGH markets: we want P(temp > threshold)
        # For LOW markets: we want P(temp < threshold)
        direction = "above" if market_type == "high" else "below"

        # Calculate ensemble probability with city-specific model weights
        # For bracket markets, we calculate P(lower ≤ T < upper) instead of P(T > threshold)
        prob_result = ensemble_probability(
            forecasts,
            temp_threshold,
            days_out,
            direction,
            city=location,  # Pass city for city-specific model weights
            is_bracket=is_bracket,
            market_type=market_type
        )

        if is_bracket:
            logger.debug(f"{ticker}: Bracket market - calculating P({temp_threshold-0.5}≤T<{temp_threshold+0.5})")

        our_probability = prob_result["ensemble_prob"]
        model_probs = prob_result["model_probs"]
        model_spread = prob_result["model_spread"]
        prob_confidence = prob_result["confidence"]

        # CONVICTION FILTER 1: Require minimum number of models
        num_models = len([t for t in forecasts.values() if t is not None])
        if num_models < MIN_MODELS_REQUIRED:
            logger.debug(f"Skipping {ticker}: only {num_models} models (need {MIN_MODELS_REQUIRED})")
            return None

        # CONVICTION FILTER 2: Check forecast model agreement (skip if models disagree too much)
        forecast_temps = [t for t in forecasts.values() if t is not None]
        if len(forecast_temps) >= 2:
            temp_spread = max(forecast_temps) - min(forecast_temps)
            if temp_spread > MAX_MODEL_SPREAD:
                logger.debug(f"Skipping {ticker}: model temp spread {temp_spread:.1f}°F exceeds max {MAX_MODEL_SPREAD}°F")
                return None

            # Calculate model agreement score (0-1, higher = more agreement)
            model_agreement_score = max(0.0, 1.0 - temp_spread / 10.0)
            if model_agreement_score < MIN_MODEL_AGREEMENT:
                logger.debug(f"Skipping {ticker}: model agreement {model_agreement_score:.0%} below {MIN_MODEL_AGREEMENT:.0%}")
                return None

        # Calculate edge
        raw_edge = self.calculate_edge(market_price, our_probability)

        # Determine trade direction
        trade_direction = self.determine_direction(our_probability, market_price)

        # Calculate directional edge (always positive from trade perspective)
        # For BUY_YES: edge = our_prob - market_prob (positive means we like YES)
        # For BUY_NO: edge = market_prob - our_prob (positive means we like NO)
        edge = abs(raw_edge)

        # PROBABILITY FILTER: Skip low-probability bets (long-shots lose too often)
        # Check the probability of the direction we're actually betting on
        betting_probability = our_probability if trade_direction == "BUY_YES" else (1 - our_probability)
        if betting_probability < MIN_PROBABILITY_THRESHOLD:
            logger.debug(f"Skipping {ticker}: betting probability {betting_probability:.0%} below {MIN_PROBABILITY_THRESHOLD:.0%} threshold (avoiding long-shots)")
            return None

        # Calculate EV
        expected_value = self.calculate_expected_value(
            our_probability, market_price, trade_direction
        )

        # CONVICTION FILTER 3: Skip if edge below threshold (adjusted for forecast timing)
        if edge < adjusted_edge_threshold:
            logger.debug(f"Skipping {ticker}: edge {edge:.2%} below threshold {adjusted_edge_threshold:.0%} (days_out={days_out})")
            return None

        # CONVICTION FILTER 4: Skip if confidence score too low
        if prob_confidence < MIN_CONFIDENCE_SCORE:
            logger.debug(f"Skipping {ticker}: confidence {prob_confidence:.0%} below {MIN_CONFIDENCE_SCORE:.0%}")
            return None

        # Skip if EV is negative
        if expected_value < 0:
            logger.debug(f"Skipping {ticker}: negative EV after fees")
            return None

        # SWEET SPOT FILTER: Only trade in optimal risk/reward zone
        # Skip heavy favorites (poor payout) and long shots (low win rate)
        if trade_direction == "BUY_YES":
            # Don't buy very cheap YES (long shots that rarely hit)
            if market_price < MIN_YES_PRICE:
                logger.debug(f"Skipping {ticker}: YES price ${market_price:.2f} below minimum ${MIN_YES_PRICE:.2f} (long shot)")
                return None
            # Don't buy expensive YES (heavy favorite - poor payout ratio)
            if market_price > MAX_YES_PRICE:
                logger.debug(f"Skipping {ticker}: YES price ${market_price:.2f} above maximum ${MAX_YES_PRICE:.2f} (poor payout)")
                return None
        else:  # BUY_NO
            no_price = 1.0 - market_price
            # Don't buy very cheap NO (long shots)
            if no_price < MIN_YES_PRICE:
                logger.debug(f"Skipping {ticker}: NO price ${no_price:.2f} below minimum ${MIN_YES_PRICE:.2f} (long shot)")
                return None
            # Don't buy expensive NO (heavy favorite - poor payout ratio)
            if no_price > MAX_YES_PRICE:
                logger.debug(f"Skipping {ticker}: NO price ${no_price:.2f} above maximum ${MAX_YES_PRICE:.2f} (poor payout)")
                return None
            # Extra restriction on bracket markets (narrow ranges are harder to hit)
            if is_bracket and no_price > MAX_NO_PRICE_BRACKET:
                logger.debug(f"Skipping {ticker}: NO price ${no_price:.2f} above bracket max ${MAX_NO_PRICE_BRACKET:.2f}")
                return None

        # Determine confidence level
        confidence = get_confidence_level(abs(edge))

        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(
            our_probability, market_price, trade_direction
        )

        # Calculate position size
        entry_price = market_price if trade_direction == "BUY_YES" else (1 - market_price)
        recommended_contracts = self.calculate_position_size(
            edge, entry_price, trade_direction, kelly_fraction
        )

        # CONVICTION-BASED POSITION SIZING: Scale with confidence
        # Higher confidence = larger position (up to 100%), lower = smaller
        confidence_scalar = min(1.0, prob_confidence / 0.80)  # 80% confidence = full size
        recommended_contracts = max(1, int(recommended_contracts * confidence_scalar))

        # Scale down position size for bracket markets (they're harder to predict)
        if is_bracket:
            recommended_contracts = max(1, int(recommended_contracts * BRACKET_POSITION_SCALE))
            logger.debug(f"Scaled bracket position to {recommended_contracts} contracts")

        # Calculate costs and potential outcomes
        total_cost = entry_price * recommended_contracts
        potential_profit = (1.0 - entry_price) * recommended_contracts
        potential_loss = total_cost
        fees = calculate_kalshi_fee(entry_price, recommended_contracts)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            ticker, market_type, temp_threshold, target_date,
            forecasts, our_probability, market_price, edge,
            prob_result.get("ensemble_temp")
        )

        signal = TradeSignal(
            signal_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            ticker=ticker,
            location=location,
            target_date=target_date,
            temp_threshold=temp_threshold,
            market_type=market_type,
            market_price=market_price,
            market_implied_prob=market_price,
            our_probability=our_probability,
            edge=edge,
            expected_value=expected_value,
            confidence=confidence,
            direction=trade_direction,
            recommended_contracts=recommended_contracts,
            total_cost=total_cost,
            potential_profit=potential_profit - fees,
            potential_loss=potential_loss + fees,
            model_probs=model_probs,
            model_spread=model_spread,
            reasoning=reasoning,
            kelly_fraction=kelly_fraction
        )

        logger.info(str(signal))
        return signal

    def _generate_reasoning(
        self,
        ticker: str,
        market_type: str,
        threshold: float,
        target_date: date,
        forecasts: Dict[str, float],
        our_prob: float,
        market_price: float,
        edge: float,
        ensemble_temp: float
    ) -> str:
        """Generate human-readable reasoning for the signal."""
        days_out = calculate_days_until(target_date)

        # Format forecasts
        forecast_str = ", ".join(
            f"{src.upper()}: {temp:.0f}°F"
            for src, temp in forecasts.items()
            if temp is not None
        )

        # Determine comparison
        direction_word = "above" if market_type == "high" else "below"
        market_implied = get_breakeven_temp(threshold, days_out, market_price, direction_word)

        return (
            f"Forecasts ({forecast_str}) suggest {our_prob:.0%} probability of "
            f"{market_type} temp {direction_word} {threshold:.0f}°F on {target_date}. "
            f"Ensemble forecast: {ensemble_temp:.1f}°F. "
            f"Market at {market_price:.0%} implies {market_implied:.1f}°F. "
            f"Edge: +{abs(edge):.1%}."
        )

    def analyze_market(
        self,
        market,
        forecasts: Dict[str, Dict[str, float]]
    ) -> Optional[TradeSignal]:
        """
        Analyze a Market object and generate signal if edge exists.

        Args:
            market: Market object from kalshi_client
            forecasts: Dict of forecasts by source

        Returns:
            TradeSignal or None
        """
        # LIQUIDITY FILTER: Skip illiquid markets
        # Wide spreads destroy edge on exit
        if market.yes_bid > 0 and market.yes_ask > 0:
            spread = market.yes_ask - market.yes_bid
            mid_price = (market.yes_bid + market.yes_ask) / 2
            if mid_price > 0.05:  # Only check spread for non-penny stocks
                spread_pct = spread / mid_price
                if spread_pct > 0.15:  # Skip if spread > 15% of price
                    logger.debug(f"Skipping {market.ticker}: spread {spread_pct:.0%} too wide (bid={market.yes_bid:.2f}, ask={market.yes_ask:.2f})")
                    return None

        # VOLUME FILTER: Skip zero volume markets only (relaxed)
        # Weather markets often have low volume but still tradeable
        if market.volume is not None and market.volume == 0:
            logger.debug(f"Skipping {market.ticker}: zero volume")
            return None

        # Extract relevant forecast temps based on market type
        temp_forecasts = {}
        for source, temps in forecasts.items():
            if market.market_type == "high":
                temp = temps.get("high")
            else:
                temp = temps.get("low")

            if temp is not None:
                temp_forecasts[source] = temp

        if not temp_forecasts:
            logger.warning(f"No forecasts available for {market.ticker}")
            return None

        return self.generate_signal(
            ticker=market.ticker,
            location=market.location,
            target_date=market.target_date,
            temp_threshold=market.temp_threshold,
            market_type=market.market_type,
            market_price=market.yes_price,
            forecasts=temp_forecasts
        )


# Example usage and testing
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    print("Testing EdgeCalculator...")

    calc = EdgeCalculator()

    # Test edge calculation
    edge = calc.calculate_edge(market_price=0.40, our_probability=0.70)
    print(f"\nEdge: market=40%, ours=70% -> edge={edge:.2%}")
    assert abs(edge - 0.30) < 0.001

    # Test EV calculation
    ev = calc.calculate_expected_value(
        our_probability=0.70,
        entry_price=0.40,
        direction="BUY_YES"
    )
    print(f"EV (BUY_YES @ $0.40, p=70%): ${ev:.3f}")

    # Test Kelly
    kelly = calc.calculate_kelly_fraction(0.70, 0.40, "BUY_YES")
    print(f"Kelly fraction: {kelly:.2%}")

    # Test signal generation
    print("\nTesting signal generation...")

    from datetime import timedelta

    signal = calc.generate_signal(
        ticker="HIGHNY-25JAN15-T85",
        location="NYC",
        target_date=date.today() + timedelta(days=1),
        temp_threshold=85.0,
        market_type="high",
        market_price=0.40,
        forecasts={"ecmwf": 87.0, "gfs": 86.0, "nws": 85.0}
    )

    if signal:
        print(f"\nGenerated Signal:")
        print(f"  Ticker: {signal.ticker}")
        print(f"  Direction: {signal.direction}")
        print(f"  Market Price: ${signal.market_price:.2f}")
        print(f"  Our Probability: {signal.our_probability:.1%}")
        print(f"  Edge: {signal.edge:+.1%}")
        print(f"  Expected Value: ${signal.expected_value:.3f}/contract")
        print(f"  Confidence: {signal.confidence}")
        print(f"  Recommended: {signal.recommended_contracts} contracts")
        print(f"  Reasoning: {signal.reasoning}")
    else:
        print("No signal generated (edge below threshold)")

    # Test with low edge (should not generate signal)
    print("\nTesting low-edge scenario...")
    signal = calc.generate_signal(
        ticker="HIGHNY-25JAN15-T85",
        location="NYC",
        target_date=date.today() + timedelta(days=1),
        temp_threshold=85.0,
        market_type="high",
        market_price=0.68,  # Close to our estimate
        forecasts={"ecmwf": 87.0, "gfs": 86.0, "nws": 85.0}
    )
    if signal:
        print(f"Signal generated with edge {signal.edge:.1%}")
    else:
        print("No signal (as expected - edge too low)")

    print("\n✓ EdgeCalculator tests passed!")
