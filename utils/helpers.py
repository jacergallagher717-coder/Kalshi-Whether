"""
Utility functions for the Weather Kalshi Paper Trading System.
"""

import re
from datetime import datetime, date, timedelta
from typing import Optional, Tuple
import pytz


def parse_kalshi_ticker(ticker: str) -> dict:
    """
    Parse a Kalshi weather market ticker to extract components.

    Ticker format: HIGHNY-25JAN15-T35
    - HIGHNY = NYC high temp market
    - 25JAN15 = January 15, 2025
    - T35 = Temperature threshold 35°F

    Args:
        ticker: Kalshi market ticker string

    Returns:
        Dictionary with parsed components:
        - location: Location code (e.g., "NYC")
        - market_type: "high" or "low"
        - target_date: date object
        - temp_threshold: float temperature in °F
    """
    # Pattern: PREFIX-YYMMMDD-TTEMP
    pattern = r'^(HIGH|LOW)([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})-T(-?\d+)$'
    match = re.match(pattern, ticker)

    if not match:
        raise ValueError(f"Invalid ticker format: {ticker}")

    market_type = match.group(1).lower()  # "high" or "low"
    location_suffix = match.group(2)  # "NY", "CHI", etc.
    year = int(match.group(3)) + 2000  # "25" -> 2025
    month_str = match.group(4)  # "JAN"
    day = int(match.group(5))  # "15"
    temp = int(match.group(6))  # "35" or "-10"

    # Convert month string to number
    months = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    month = months.get(month_str.upper())
    if not month:
        raise ValueError(f"Invalid month in ticker: {month_str}")

    # Map location suffix to location code
    location_map = {
        'NY': 'NYC',
        'CHI': 'CHI',
        'LA': 'LA',
        'MIA': 'MIA'
    }
    location = location_map.get(location_suffix, location_suffix)

    return {
        'ticker': ticker,
        'location': location,
        'market_type': market_type,
        'target_date': date(year, month, day),
        'temp_threshold': float(temp)
    }


def format_kalshi_ticker(location: str, market_type: str, target_date: date, temp: int) -> str:
    """
    Format components into a Kalshi ticker string.

    Args:
        location: Location code (e.g., "NYC")
        market_type: "high" or "low"
        target_date: Target settlement date
        temp: Temperature threshold in °F

    Returns:
        Formatted ticker string
    """
    # Map location to suffix
    location_suffix_map = {
        'NYC': 'NY',
        'CHI': 'CHI',
        'LA': 'LA',
        'MIA': 'MIA'
    }
    suffix = location_suffix_map.get(location, location)

    prefix = f"{market_type.upper()}{suffix}"
    date_str = target_date.strftime("%y%b%d").upper()

    return f"{prefix}-{date_str}-T{temp}"


def calculate_days_until(target_date: date, allow_negative: bool = False) -> int:
    """
    Calculate days until target date from today.

    Args:
        target_date: Target date
        allow_negative: If True, return negative days for past dates

    Returns:
        Number of days (0 = today, 1 = tomorrow, negative for past if allowed)
    """
    today = date.today()
    delta = target_date - today
    if allow_negative:
        return delta.days
    return max(0, delta.days)


def get_eastern_time() -> datetime:
    """Get current time in Eastern timezone."""
    eastern = pytz.timezone('America/New_York')
    return datetime.now(eastern)


def is_market_hours() -> bool:
    """
    Check if Kalshi markets are currently open.

    Kalshi weather markets are typically open 24/7 but may have
    reduced liquidity outside of normal hours.

    Returns:
        True if within active trading hours (8 AM - 10 PM ET)
    """
    et_now = get_eastern_time()
    return 8 <= et_now.hour < 22


def price_to_probability(price: float) -> float:
    """
    Convert market price to implied probability.

    Args:
        price: Market price (0-1 range)

    Returns:
        Implied probability (0-1 range)
    """
    return max(0.0, min(1.0, price))


def probability_to_price(prob: float) -> float:
    """
    Convert probability to expected market price.

    Args:
        prob: Probability (0-1 range)

    Returns:
        Expected price (0-1 range)
    """
    return max(0.0, min(1.0, prob))


def calculate_pnl(entry_price: float, exit_price: float, contracts: int,
                  direction: str, fees: float = 0.0) -> Tuple[float, float]:
    """
    Calculate profit/loss for a trade.

    Args:
        entry_price: Entry price (0-1)
        exit_price: Exit price (0-1), typically 0 or 1 at settlement
        contracts: Number of contracts
        direction: "BUY_YES" or "BUY_NO"
        fees: Total fees paid

    Returns:
        Tuple of (gross_pnl, net_pnl)
    """
    if direction == "BUY_YES":
        # Bought YES: profit if exit > entry
        gross_pnl = (exit_price - entry_price) * contracts
    else:  # BUY_NO
        # Bought NO: profit if exit < entry (YES goes down)
        gross_pnl = (entry_price - exit_price) * contracts

    net_pnl = gross_pnl - fees
    return gross_pnl, net_pnl


def format_currency(amount: float) -> str:
    """Format a dollar amount for display."""
    if amount >= 0:
        return f"${amount:.2f}"
    else:
        return f"-${abs(amount):.2f}"


def format_percent(value: float) -> str:
    """Format a percentage for display."""
    return f"{value * 100:.1f}%"
