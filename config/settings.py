"""
Configuration settings for the Weather Kalshi Paper Trading System.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Kalshi API Configuration
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY", "")
KALSHI_API_SECRET = os.getenv("KALSHI_API_SECRET", "")
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")
KALSHI_EMAIL = os.getenv("KALSHI_EMAIL", "")
KALSHI_PASSWORD = os.getenv("KALSHI_PASSWORD", "")

# Visual Crossing Weather API
VISUALCROSSING_API_KEY = os.getenv("VISUALCROSSING_API_KEY", "")

# Use demo API by default for safety (set KALSHI_USE_DEMO=false for production)
KALSHI_USE_DEMO = os.getenv("KALSHI_USE_DEMO", "true").lower() == "true"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_PUBLIC_URL = "https://api.elections.kalshi.com/trade-api/v2"  # Public API - no auth required
KALSHI_BASE_URL = KALSHI_DEMO_URL if KALSHI_USE_DEMO else KALSHI_PROD_URL

# Auto-trading Configuration
AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"
AUTO_TRADE_MAX_DAILY_TRADES = int(os.getenv("AUTO_TRADE_MAX_DAILY_TRADES", "999"))  # No limit - conviction is the filter
AUTO_TRADE_MAX_OPEN_POSITIONS = int(os.getenv("AUTO_TRADE_MAX_OPEN_POSITIONS", "999"))  # No limit - conviction is the filter

# Trading Parameters - BALANCED (based on Jan 5-6 trade analysis)
# Philly won big with ~40% edge, but we don't want to miss good opportunities
MIN_EDGE_THRESHOLD = float(os.getenv("MIN_EDGE_THRESHOLD", "0.30"))  # 30% minimum edge
MIN_PROBABILITY_THRESHOLD = float(os.getenv("MIN_PROBABILITY_THRESHOLD", "0.20"))  # 20% min probability (skip long-shots)
MIN_CONFIDENCE_SCORE = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.50"))  # 50% confidence required
MIN_MODEL_AGREEMENT = float(os.getenv("MIN_MODEL_AGREEMENT", "0.65"))  # 65% model agreement
MAX_POSITION_SIZE = int(os.getenv("MAX_POSITION_SIZE", "225"))  # Max $2.25 per trade (1.5x increase)
MAX_CONTRACTS_PER_TRADE = int(os.getenv("MAX_CONTRACTS_PER_TRADE", "30"))  # Max 30 contracts (1.5x increase)

# Aggressive Sizing for High-Edge Trades
# When edge is 50%+, we're very confident - size up
HIGH_EDGE_THRESHOLD = float(os.getenv("HIGH_EDGE_THRESHOLD", "0.50"))  # 50%+ edge = high conviction
HIGH_EDGE_POSITION_PERCENT = float(os.getenv("HIGH_EDGE_POSITION_PERCENT", "0.075"))  # 7.5% of bankroll for 50%+ edge (1.5x)
NORMAL_POSITION_PERCENT = float(os.getenv("NORMAL_POSITION_PERCENT", "0.03"))  # 3% of bankroll for normal trades (1.5x)

# Time-based Trading Filters (forecast accuracy decays with time)
# Forecasts update overnight - trading tomorrow's weather at midnight is risky
MAX_FORECAST_DAYS = int(os.getenv("MAX_FORECAST_DAYS", "1"))  # Only trade same-day (0) and next-day (1)
NEXT_DAY_EDGE_PENALTY = float(os.getenv("NEXT_DAY_EDGE_PENALTY", "0.10"))  # Require 10% more edge for next-day
PREFER_SAME_DAY = os.getenv("PREFER_SAME_DAY", "true").lower() == "true"  # Prioritize same-day markets

# Morning Trading Window (times in Eastern)
# Best time to trade: 6-9am ET when overnight model runs have settled but market hasn't adjusted
TRADING_WINDOW_ENABLED = os.getenv("TRADING_WINDOW_ENABLED", "true").lower() == "true"
TRADING_WINDOW_START_HOUR = int(os.getenv("TRADING_WINDOW_START_HOUR", "6"))  # 6am ET
TRADING_WINDOW_END_HOUR = int(os.getenv("TRADING_WINDOW_END_HOUR", "21"))  # 9pm ET (allow evening trades)
OVERNIGHT_EDGE_PENALTY = float(os.getenv("OVERNIGHT_EDGE_PENALTY", "0.15"))  # 15% more edge required 9pm-6am

# Price Filters
MIN_YES_PRICE = float(os.getenv("MIN_YES_PRICE", "0.10"))  # Don't buy YES below 10 cents (was 8)
MAX_NO_PRICE_BRACKET = float(os.getenv("MAX_NO_PRICE_BRACKET", "0.45"))  # Don't buy NO above 45 cents
BRACKET_POSITION_SCALE = float(os.getenv("BRACKET_POSITION_SCALE", "0.5"))  # Half position size on brackets
MAX_MODEL_SPREAD = float(os.getenv("MAX_MODEL_SPREAD", "3.0"))  # 3°F spread allowed (balanced)
MIN_MODELS_REQUIRED = int(os.getenv("MIN_MODELS_REQUIRED", "3"))  # Require at least 3 models

CONFIDENCE_LEVELS = {
    "high": 0.15,              # 15%+ edge = high confidence
    "medium": 0.10,            # 10-15% edge = medium confidence
    "low": 0.05                # 5-10% edge = low confidence (no trade)
}

# Model Weights for Ensemble (4 proven sources for US weather)
MODEL_WEIGHTS = {
    "ecmwf": 0.35,           # European model - most accurate globally
    "gfs": 0.20,             # US NOAA - good for North America short term
    "nws": 0.25,             # Official US forecast - likely used for Kalshi settlement
    "visualcrossing": 0.20   # Visual Crossing - commercial accuracy
}

# City-Specific Model Weights (override defaults based on historical accuracy)
# Format: city -> {model: weight}
# These will be auto-updated by the model tracker as we gather data
CITY_MODEL_WEIGHTS = {
    # US East Coast - ECMWF and NWS are strong
    "NYC": {"ecmwf": 0.35, "gfs": 0.15, "nws": 0.30, "visualcrossing": 0.20},
    "PHI": {"ecmwf": 0.35, "gfs": 0.15, "nws": 0.30, "visualcrossing": 0.20},
    # US Midwest - GFS performs better here
    "CHI": {"ecmwf": 0.30, "gfs": 0.25, "nws": 0.25, "visualcrossing": 0.20},
    # US West Coast - Marine influence, ECMWF strong
    "LA": {"ecmwf": 0.40, "gfs": 0.15, "nws": 0.25, "visualcrossing": 0.20},
    # Florida - Tropical, ECMWF strong
    "MIA": {"ecmwf": 0.40, "gfs": 0.15, "nws": 0.25, "visualcrossing": 0.20},
    # Texas - Continental, GFS good
    "AUS": {"ecmwf": 0.30, "gfs": 0.25, "nws": 0.25, "visualcrossing": 0.20},
    # Colorado - Mountain weather, balanced approach
    "DEN": {"ecmwf": 0.30, "gfs": 0.25, "nws": 0.25, "visualcrossing": 0.20},
}
USE_CITY_WEIGHTS = os.getenv("USE_CITY_WEIGHTS", "true").lower() == "true"

# Temperature Probability Distribution
# Standard deviation for temperature forecasts (in °F)
# Increases with forecast horizon
TEMP_UNCERTAINTY = {
    0: 1.0,   # Same day: ±1°F std dev (forecasts very accurate)
    1: 2.0,   # 1 day out: ±2°F std dev
    2: 3.0,   # 2 days out: ±3°F std dev
    3: 4.0,   # 3 days out: ±4°F std dev
    4: 5.0,   # 4 days out: ±5°F std dev
    5: 6.0,   # 5 days out: ±6°F std dev
    6: 7.0,   # 6 days out: ±7°F std dev
    7: 8.0    # 7 days out: ±8°F std dev
}

# Data Collection Schedule
COLLECTION_INTERVAL_MINUTES = 60  # Collect data every hour
TRADING_CHECK_INTERVAL_MINUTES = 15  # Check for trades every 15 min

# Database Configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/storage/")
FORECASTS_DB = os.path.join(DATABASE_PATH, "forecasts.db")
TRADES_DB = os.path.join(DATABASE_PATH, "trades.db")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = "./logs/"


def calculate_kalshi_fee(price: float, contracts: int) -> float:
    """
    Calculate Kalshi trading fee.

    Fee = 0.07 * contracts * price * (1 - price)
    Max fee = $0.0175 per contract (at 50¢)

    Args:
        price: Contract price (0-1)
        contracts: Number of contracts

    Returns:
        Total fee in dollars
    """
    fee_per_contract = 0.07 * price * (1 - price)
    fee_per_contract = min(fee_per_contract, 0.0175)
    return fee_per_contract * contracts


def get_confidence_level(edge: float) -> str:
    """
    Determine confidence level based on edge.

    Args:
        edge: Calculated edge (our probability - market probability)

    Returns:
        Confidence level string: "high", "medium", or "low"
    """
    if edge >= CONFIDENCE_LEVELS["high"]:
        return "high"
    elif edge >= CONFIDENCE_LEVELS["medium"]:
        return "medium"
    else:
        return "low"
