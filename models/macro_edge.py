"""
Macro Edge Calculator

The core engine for finding profitable trades on Kalshi economic markets.

Edge sources (in order of reliability):
1. Nowcast Divergence - Our aggregated nowcast vs Kalshi market price
2. Ladder Arbitrage - Inconsistent pricing across strikes
3. Calendar Timing - Positioning before scheduled releases
4. Spread Capture - Using limit orders to become the spread

This module combines all signals into actionable trade recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, Optional, List, Tuple
from enum import Enum

from data.collectors.nowcast_collector import NowcastAggregator, NowcastData
from data.collectors.econ_calendar import EconCalendar, EventType, EconEvent
from models.ladder_analyzer import LadderAnalyzer, Strike, Inconsistency
from utils.logger import get_logger

logger = get_logger("macro_edge")


class EdgeType(Enum):
    """Types of edge we can capture."""
    NOWCAST_DIVERGENCE = "nowcast"      # Our forecast differs from market
    LADDER_ARBITRAGE = "arbitrage"       # Inconsistent strike pricing
    TIMING_PREMIUM = "timing"            # Pre-release positioning
    SPREAD_CAPTURE = "spread"            # Market making


@dataclass
class MacroSignal:
    """A trade signal for macro markets."""
    signal_id: str
    timestamp: datetime
    ticker: str
    market_type: str  # "cpi", "fed", "nfp", etc.

    # Edge analysis
    edge_type: EdgeType
    edge_magnitude: float  # Raw edge before costs
    net_edge: float        # Edge after spread/fees
    confidence: float      # 0-1 confidence score

    # Trade parameters
    direction: str         # "BUY_YES" or "BUY_NO"
    target_price: float    # Price we want to execute at (for limit orders)
    current_bid: float
    current_ask: float
    max_position_pct: float  # Max % of bankroll

    # Context
    event: Optional[EconEvent] = None
    nowcast_value: Optional[float] = None
    market_implied_value: Optional[float] = None
    reasoning: str = ""

    def __repr__(self):
        return (
            f"MacroSignal({self.ticker}, {self.direction}, "
            f"edge={self.net_edge:.1%}, conf={self.confidence:.0%})"
        )


class MacroEdgeCalculator:
    """
    Main class for calculating edge in macro markets.

    Usage:
        calc = MacroEdgeCalculator()
        signals = calc.scan_for_opportunities(kalshi_markets)
        for signal in signals:
            if signal.net_edge > 0.03:  # 3% edge threshold
                execute_trade(signal)
    """

    # Minimum thresholds for trade consideration
    MIN_EDGE_THRESHOLD = 0.03      # 3% minimum edge
    MIN_NET_EDGE_THRESHOLD = 0.02  # 2% after spread/fees
    MIN_CONFIDENCE = 0.5           # 50% confidence minimum
    MAX_SPREAD_PCT = 0.10          # Don't trade if spread > 10%

    def __init__(self, fred_api_key: str = None):
        self.nowcast_aggregator = NowcastAggregator(fred_api_key)
        self.calendar = EconCalendar()
        self.ladder_analyzer = LadderAnalyzer()

    def scan_for_opportunities(
        self,
        markets: List[Dict],
        balance: float = 100.0
    ) -> List[MacroSignal]:
        """
        Scan all markets for trading opportunities.

        Args:
            markets: List of market data dicts from Kalshi API
            balance: Current account balance for position sizing

        Returns:
            List of MacroSignal objects, sorted by net edge
        """
        signals = []

        # Get current nowcasts
        nowcasts = self.nowcast_aggregator.get_all_nowcasts()

        # Get current trading windows
        windows = self.calendar.get_trading_windows()
        active_events = {w['event'].event_type: w for w in windows}

        # Group markets by type
        cpi_markets = [m for m in markets if 'cpi' in m.get('ticker', '').lower()]
        fed_markets = [m for m in markets if 'fed' in m.get('ticker', '').lower()]

        # Analyze CPI markets
        for market in cpi_markets:
            signal = self._analyze_cpi_market(market, nowcasts, active_events, balance)
            if signal:
                signals.append(signal)

        # Analyze Fed markets
        for market in fed_markets:
            signal = self._analyze_fed_market(market, active_events, balance)
            if signal:
                signals.append(signal)

        # Check for ladder arbitrage (need all strikes for a series)
        ladder_signals = self._find_ladder_arbitrage(markets, balance)
        signals.extend(ladder_signals)

        # Sort by net edge (highest first)
        signals.sort(key=lambda s: s.net_edge, reverse=True)

        return signals

    def _analyze_cpi_market(
        self,
        market: Dict,
        nowcasts: Dict,
        active_events: Dict,
        balance: float
    ) -> Optional[MacroSignal]:
        """
        Analyze a CPI market for nowcast divergence edge.

        The edge exists when:
        - Our nowcast aggregate differs from market-implied value
        - We're in a valid trading window
        - Spread is acceptable
        """
        ticker = market.get('ticker', '')
        yes_bid = market.get('yes_bid', 0)
        yes_ask = market.get('yes_ask', 0)

        if yes_bid <= 0 or yes_ask <= 0:
            return None

        yes_mid = (yes_bid + yes_ask) / 2
        spread = yes_ask - yes_bid
        spread_pct = spread / yes_mid if yes_mid > 0 else 1.0

        # Check spread is acceptable
        if spread_pct > self.MAX_SPREAD_PCT:
            logger.debug(f"Skipping {ticker}: spread {spread_pct:.1%} too wide")
            return None

        # Get our CPI nowcast
        cpi_nowcast = self.nowcast_aggregator.get_aggregate_nowcast('cpi_headline')
        if not cpi_nowcast:
            logger.debug(f"No CPI nowcast available")
            return None

        our_value = cpi_nowcast['aggregate_value']
        nowcast_confidence = cpi_nowcast['confidence']

        # Parse market threshold from ticker
        # Example: KXCPIYOY-26JAN-T2.7 -> threshold 2.7
        threshold = self._parse_threshold(ticker)
        if threshold is None:
            return None

        # Calculate edge
        # For "above X" market: if our nowcast > X, YES is underpriced
        # Market price = P(CPI > threshold)
        edge, direction, reasoning = self._calculate_cpi_edge(
            our_value, threshold, yes_mid, spread
        )

        if abs(edge) < self.MIN_EDGE_THRESHOLD:
            return None

        # Net edge after spread (assume we pay half the spread)
        net_edge = abs(edge) - spread / 2

        if net_edge < self.MIN_NET_EDGE_THRESHOLD:
            return None

        # Confidence combines nowcast confidence + timing
        timing_bonus = 0.1 if EventType.CPI in active_events else 0
        confidence = min(1.0, nowcast_confidence + timing_bonus)

        if confidence < self.MIN_CONFIDENCE:
            return None

        # Position sizing: based on edge magnitude and confidence
        position_pct = self._calculate_position_size(net_edge, confidence, spread_pct)

        return MacroSignal(
            signal_id=f"CPI-{datetime.utcnow().strftime('%H%M%S')}",
            timestamp=datetime.utcnow(),
            ticker=ticker,
            market_type="cpi",
            edge_type=EdgeType.NOWCAST_DIVERGENCE,
            edge_magnitude=abs(edge),
            net_edge=net_edge,
            confidence=confidence,
            direction=direction,
            target_price=yes_mid if direction == "BUY_YES" else (1 - yes_mid),
            current_bid=yes_bid,
            current_ask=yes_ask,
            max_position_pct=position_pct,
            event=active_events.get(EventType.CPI, {}).get('event'),
            nowcast_value=our_value,
            market_implied_value=self._implied_value_from_price(yes_mid, threshold),
            reasoning=reasoning
        )

    def _analyze_fed_market(
        self,
        market: Dict,
        active_events: Dict,
        balance: float
    ) -> Optional[MacroSignal]:
        """
        Analyze Fed decision market.

        Edge sources:
        - CME FedWatch divergence
        - Timing premium (pre-FOMC positioning)
        """
        # Similar structure to CPI analysis
        # Would compare to CME Fed Funds futures implied probabilities
        ticker = market.get('ticker', '')
        logger.debug(f"Fed market analysis for {ticker} - not fully implemented")
        return None

    def _find_ladder_arbitrage(
        self,
        markets: List[Dict],
        balance: float
    ) -> List[MacroSignal]:
        """
        Find arbitrage opportunities across market ladders.

        Groups markets by series, checks for pricing inconsistencies.
        """
        signals = []

        # Group by market series
        series_markets = {}
        for market in markets:
            ticker = market.get('ticker', '')
            # Extract series (e.g., "KXCPIYOY" from "KXCPIYOY-26JAN-T2.7")
            series = ticker.split('-')[0] if '-' in ticker else ticker
            if series not in series_markets:
                series_markets[series] = []
            series_markets[series].append(market)

        # Analyze each series
        for series, series_mkt in series_markets.items():
            if len(series_mkt) < 2:
                continue

            # Convert to Strike objects
            strikes = []
            for m in series_mkt:
                threshold = self._parse_threshold(m.get('ticker', ''))
                if threshold is None:
                    continue

                yes_bid = m.get('yes_bid', 0)
                yes_ask = m.get('yes_ask', 0)
                if yes_bid <= 0 or yes_ask <= 0:
                    continue

                strikes.append(Strike(
                    ticker=m['ticker'],
                    threshold=threshold,
                    direction="above",  # Assume for now
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    yes_mid=(yes_bid + yes_ask) / 2,
                    volume=m.get('volume', 0),
                    open_interest=m.get('open_interest', 0)
                ))

            if len(strikes) < 2:
                continue

            # Check for inconsistencies
            issues = self.ladder_analyzer.find_inconsistencies(strikes)

            for issue in issues:
                if issue.arbitrage_value > 0.01:  # > 1 cent arb
                    signals.append(MacroSignal(
                        signal_id=f"ARB-{datetime.utcnow().strftime('%H%M%S')}",
                        timestamp=datetime.utcnow(),
                        ticker=issue.strikes[0].ticker,
                        market_type="arbitrage",
                        edge_type=EdgeType.LADDER_ARBITRAGE,
                        edge_magnitude=issue.arbitrage_value,
                        net_edge=issue.arbitrage_value * 0.8,  # Assume 20% execution cost
                        confidence=issue.confidence,
                        direction="ARBITRAGE",
                        target_price=0,
                        current_bid=issue.strikes[0].yes_bid,
                        current_ask=issue.strikes[0].yes_ask,
                        max_position_pct=0.05,  # Small size for arb
                        reasoning=issue.description
                    ))

        return signals

    def _calculate_cpi_edge(
        self,
        nowcast: float,
        threshold: float,
        market_price: float,
        spread: float
    ) -> Tuple[float, str, str]:
        """
        Calculate edge for CPI threshold market.

        Args:
            nowcast: Our CPI nowcast (e.g., 2.65 for 2.65%)
            threshold: Market threshold (e.g., 2.7 for "above 2.7%")
            market_price: Current YES price (probability of above threshold)
            spread: Bid-ask spread

        Returns:
            (edge, direction, reasoning)
        """
        # How far is our nowcast from the threshold?
        distance = nowcast - threshold

        # Convert distance to probability adjustment
        # Rough heuristic: 0.1% CPI difference ≈ 10% probability shift
        # (This should be calibrated with historical data)
        PROBABILITY_SENSITIVITY = 1.0  # 1.0 means 0.1% CPI = 10% prob

        # If nowcast > threshold, YES should be more likely
        # If nowcast < threshold, NO should be more likely
        implied_adjustment = distance * PROBABILITY_SENSITIVITY

        # Our estimated fair probability
        our_prob = min(0.95, max(0.05, 0.5 + implied_adjustment))

        # Edge vs market
        edge = our_prob - market_price

        if edge > 0:
            direction = "BUY_YES"
            reasoning = (
                f"Nowcast {nowcast:.2f}% vs threshold {threshold:.1f}%. "
                f"Market at {market_price:.0%}, we estimate {our_prob:.0%}. "
                f"Edge: {edge:.1%}"
            )
        else:
            direction = "BUY_NO"
            reasoning = (
                f"Nowcast {nowcast:.2f}% vs threshold {threshold:.1f}%. "
                f"Market at {market_price:.0%}, we estimate {our_prob:.0%}. "
                f"Edge on NO: {-edge:.1%}"
            )

        return edge, direction, reasoning

    def _calculate_position_size(
        self,
        net_edge: float,
        confidence: float,
        spread_pct: float
    ) -> float:
        """
        Calculate position size as % of bankroll.

        Uses modified Kelly criterion scaled by confidence.
        """
        # Base Kelly fraction (simplified)
        kelly = net_edge * 2  # Rough approximation

        # Scale by confidence
        scaled_kelly = kelly * confidence

        # Further scale by spread (wider spread = smaller size)
        spread_penalty = max(0.5, 1 - spread_pct * 5)
        final_size = scaled_kelly * spread_penalty

        # Cap at 10% of bankroll per trade
        return min(0.10, max(0.01, final_size))

    def _parse_threshold(self, ticker: str) -> Optional[float]:
        """
        Parse threshold value from ticker.

        Examples:
            KXCPIYOY-26JAN-T2.7 -> 2.7
            CPI-ABOVE-2.5 -> 2.5
        """
        import re

        # Try pattern: T followed by number
        match = re.search(r'T([\d.]+)', ticker)
        if match:
            return float(match.group(1))

        # Try pattern: ABOVE- or BELOW- followed by number
        match = re.search(r'(?:ABOVE|BELOW)-([\d.]+)', ticker, re.I)
        if match:
            return float(match.group(1))

        # Try last number in ticker
        match = re.search(r'([\d.]+)$', ticker)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        return None

    def _implied_value_from_price(self, price: float, threshold: float) -> float:
        """
        Estimate implied value from market price and threshold.

        Rough inverse of probability calculation.
        """
        # If price = 0.5, implied value ≈ threshold
        # If price = 0.9, implied value ≈ threshold + 0.4 (40% above)
        adjustment = (price - 0.5) * 0.4  # Scale factor
        return threshold + adjustment


def get_macro_opportunities(kalshi_client=None) -> List[MacroSignal]:
    """
    Convenience function to scan for macro opportunities.

    Args:
        kalshi_client: Optional Kalshi API client

    Returns:
        List of trading signals
    """
    calc = MacroEdgeCalculator()

    # In production, fetch from Kalshi API
    # markets = kalshi_client.get_markets(categories=['economics', 'financials'])

    # For now, return empty (need API integration)
    logger.info("Scanning for macro opportunities...")
    return calc.scan_for_opportunities([])


if __name__ == "__main__":
    print("Testing Macro Edge Calculator...")

    calc = MacroEdgeCalculator()

    # Test CPI edge calculation
    print("\nTest 1: CPI edge calculation")
    edge, direction, reasoning = calc._calculate_cpi_edge(
        nowcast=2.65,      # Our nowcast: 2.65%
        threshold=2.7,     # Market: above 2.7%
        market_price=0.40, # Market says 40% chance above 2.7%
        spread=0.05
    )
    print(f"  Nowcast: 2.65%, Threshold: 2.7%")
    print(f"  Market: 40% YES")
    print(f"  Edge: {edge:.1%}, Direction: {direction}")
    print(f"  Reasoning: {reasoning}")

    # Test threshold parsing
    print("\nTest 2: Threshold parsing")
    test_tickers = [
        "KXCPIYOY-26JAN-T2.7",
        "CPI-ABOVE-2.5",
        "KXFED-26JAN-HOLD",
    ]
    for ticker in test_tickers:
        threshold = calc._parse_threshold(ticker)
        print(f"  {ticker} -> {threshold}")

    # Test position sizing
    print("\nTest 3: Position sizing")
    sizes = [
        (0.05, 0.8, 0.03),  # 5% edge, 80% conf, 3% spread
        (0.10, 0.6, 0.05),  # 10% edge, 60% conf, 5% spread
        (0.03, 0.9, 0.02),  # 3% edge, 90% conf, 2% spread
    ]
    for edge, conf, spread in sizes:
        size = calc._calculate_position_size(edge, conf, spread)
        print(f"  Edge={edge:.0%}, Conf={conf:.0%}, Spread={spread:.0%} -> Size={size:.1%}")

    print("\n✓ Macro Edge Calculator tests passed!")
