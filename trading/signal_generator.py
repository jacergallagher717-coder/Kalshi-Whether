"""
Signal generator that identifies trade opportunities.

Scans all available markets, compares to forecast data, and generates
trade signals when edge exceeds threshold.
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Optional

from config.locations import ACTIVE_LOCATIONS
from data.collectors import DataManager, KalshiClient, WeatherClient
from data.collectors.kalshi_client import Market
from models.edge_calculator import EdgeCalculator, TradeSignal
from models.ensemble import EnsembleModel
from utils.logger import get_logger

logger = get_logger("signal_generator")


class SignalGenerator:
    """
    Generates trading signals by analyzing market opportunities.

    Process:
    1. Fetch all active weather markets
    2. Get latest forecasts for relevant dates
    3. Calculate edge for each market
    4. Generate signals for markets with sufficient edge
    """

    def __init__(
        self,
        data_manager: DataManager = None,
        edge_calculator: EdgeCalculator = None
    ):
        """
        Initialize the signal generator.

        Args:
            data_manager: Optional data manager instance
            edge_calculator: Optional edge calculator instance
        """
        self.data_manager = data_manager or DataManager()
        self.edge_calculator = edge_calculator or EdgeCalculator()

        logger.info("Signal generator initialized")

    def scan_markets(
        self,
        locations: List[str] = None,
        max_days_out: int = 7
    ) -> List[TradeSignal]:
        """
        Scan all markets and generate signals for opportunities.

        Args:
            locations: List of location codes to scan
            max_days_out: Maximum days until settlement to consider

        Returns:
            List of TradeSignals for actionable opportunities
        """
        locations = locations or ACTIVE_LOCATIONS
        signals = []

        for location in locations:
            location_signals = self._scan_location(location, max_days_out)
            signals.extend(location_signals)

        # Sort by edge (highest first)
        signals.sort(key=lambda s: abs(s.edge), reverse=True)

        logger.info(f"Generated {len(signals)} signals from market scan")
        return signals

    def _scan_location(
        self,
        location: str,
        max_days_out: int
    ) -> List[TradeSignal]:
        """Scan markets for a single location."""
        signals = []

        # Get markets
        try:
            markets = self.data_manager.get_markets(location)
        except Exception as e:
            logger.error(f"Error fetching markets for {location}: {e}")
            return signals

        if not markets:
            logger.warning(f"No markets found for {location}")
            return signals

        logger.info(f"Scanning {len(markets)} markets for {location}")

        # Group markets by target date
        markets_by_date: Dict[date, List[Market]] = {}
        for market in markets:
            if market.target_date not in markets_by_date:
                markets_by_date[market.target_date] = []
            markets_by_date[market.target_date].append(market)

        # Process each date
        for target_date, date_markets in markets_by_date.items():
            # Skip if too far out
            days_out = (target_date - date.today()).days
            if days_out < 0 or days_out > max_days_out:
                continue

            # Get forecasts for this date
            try:
                forecasts = self._get_forecasts_for_date(location, target_date)
            except Exception as e:
                logger.error(f"Error fetching forecasts for {location} {target_date}: {e}")
                continue

            if not forecasts:
                logger.warning(f"No forecasts available for {location} {target_date}")
                continue

            # Analyze each market
            for market in date_markets:
                try:
                    signal = self.edge_calculator.analyze_market(market, forecasts)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error analyzing market {market.ticker}: {e}")
                    continue

        return signals

    def _get_forecasts_for_date(
        self,
        location: str,
        target_date: date
    ) -> Dict[str, Dict[str, float]]:
        """
        Get forecasts from all sources for a date.

        Returns dict like:
        {
            "ecmwf": {"high": 85.0, "low": 70.0},
            "gfs": {"high": 84.0, "low": 69.0},
            "nws": {"high": 83.0, "low": 68.0}
        }
        """
        # Try to get from database first
        stored_forecasts = self.data_manager.get_latest_forecasts(location, target_date)

        if stored_forecasts:
            # Convert Forecast objects to dict format
            result = {}
            for source, forecast in stored_forecasts.items():
                result[source] = {
                    "high": forecast.high_temp_f,
                    "low": forecast.low_temp_f
                }
            return result

        # Fall back to live fetch
        logger.info(f"Fetching live forecasts for {location} {target_date}")
        all_forecasts = self.data_manager.weather_client.get_forecasts_for_date(
            location, target_date
        )

        result = {}
        for source, forecast in all_forecasts.items():
            result[source] = {
                "high": forecast.high_temp_f,
                "low": forecast.low_temp_f
            }

        return result

    def generate_single_signal(
        self,
        ticker: str,
        force: bool = False
    ) -> Optional[TradeSignal]:
        """
        Generate signal for a specific market ticker.

        Args:
            ticker: Market ticker to analyze
            force: Generate signal even if edge below threshold

        Returns:
            TradeSignal or None
        """
        from utils.helpers import parse_kalshi_ticker

        # Parse ticker to get components
        try:
            parsed = parse_kalshi_ticker(ticker)
        except ValueError as e:
            logger.error(f"Invalid ticker format: {ticker}")
            return None

        location = parsed["location"]
        target_date = parsed["target_date"]
        threshold = parsed["temp_threshold"]
        market_type = parsed["market_type"]

        # Get market data
        market = self.data_manager.kalshi_client.get_market(ticker)
        if not market:
            logger.error(f"Market not found: {ticker}")
            return None

        # Get forecasts
        forecasts = self._get_forecasts_for_date(location, target_date)
        if not forecasts:
            logger.error(f"No forecasts available for {ticker}")
            return None

        # Extract relevant temps
        temp_forecasts = {}
        for source, temps in forecasts.items():
            temp = temps.get("high" if market_type == "high" else "low")
            if temp is not None:
                temp_forecasts[source] = temp

        # Generate signal
        signal = self.edge_calculator.generate_signal(
            ticker=ticker,
            location=location,
            target_date=target_date,
            temp_threshold=threshold,
            market_type=market_type,
            market_price=market.yes_price,
            forecasts=temp_forecasts
        )

        return signal

    def get_market_analysis(
        self,
        ticker: str
    ) -> Dict:
        """
        Get detailed analysis for a specific market.

        Returns comprehensive data about the market and our assessment.
        """
        from utils.helpers import parse_kalshi_ticker, calculate_days_until

        try:
            parsed = parse_kalshi_ticker(ticker)
        except ValueError:
            return {"error": f"Invalid ticker: {ticker}"}

        location = parsed["location"]
        target_date = parsed["target_date"]
        threshold = parsed["temp_threshold"]
        market_type = parsed["market_type"]
        days_out = calculate_days_until(target_date)

        # Get market data
        market = self.data_manager.kalshi_client.get_market(ticker)
        if not market:
            return {"error": f"Market not found: {ticker}"}

        # Get forecasts
        forecasts = self._get_forecasts_for_date(location, target_date)

        # Calculate probability from each model
        from models.probability import forecast_to_probability, get_breakeven_temp

        direction = "above" if market_type == "high" else "below"
        model_analysis = {}

        for source, temps in forecasts.items():
            temp = temps.get("high" if market_type == "high" else "low")
            if temp is not None:
                prob = forecast_to_probability(temp, threshold, days_out, direction)
                model_analysis[source] = {
                    "forecast_temp": temp,
                    "probability": prob
                }

        # Market implied temperature
        market_implied_temp = get_breakeven_temp(threshold, days_out, market.yes_price, direction)

        # Generate signal
        signal = self.generate_single_signal(ticker)

        return {
            "ticker": ticker,
            "location": location,
            "target_date": target_date.isoformat(),
            "days_out": days_out,
            "temp_threshold": threshold,
            "market_type": market_type,
            "market_data": {
                "yes_price": market.yes_price,
                "no_price": market.no_price,
                "yes_bid": market.yes_bid,
                "yes_ask": market.yes_ask,
                "volume": market.volume,
                "open_interest": market.open_interest
            },
            "market_implied_temp": market_implied_temp,
            "model_analysis": model_analysis,
            "signal": {
                "direction": signal.direction if signal else None,
                "edge": signal.edge if signal else None,
                "our_probability": signal.our_probability if signal else None,
                "expected_value": signal.expected_value if signal else None,
                "confidence": signal.confidence if signal else None,
                "reasoning": signal.reasoning if signal else None
            } if signal else None
        }


# Example usage
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    print("Testing SignalGenerator...")

    generator = SignalGenerator()

    # Scan markets
    print("\nScanning markets for signals...")
    signals = generator.scan_markets()

    print(f"\nFound {len(signals)} signals:")
    for signal in signals[:5]:  # Show top 5
        print(f"  {signal}")
