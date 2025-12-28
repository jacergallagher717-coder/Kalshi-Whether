"""
City configurations for weather markets.

Each location includes:
- Geographic coordinates for weather API queries
- NWS gridpoint for official forecasts
- Kalshi ticker prefixes for market lookups
- Settlement station reference
"""

LOCATIONS = {
    "NYC": {
        "name": "New York City",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timezone": "America/New_York",
        "nws_gridpoint": "OKX/33,37",
        "kalshi_high_prefix": "HIGHNY",
        "kalshi_low_prefix": "LOWNY",
        "settlement_station": "Central Park"
    },
    "CHI": {
        "name": "Chicago",
        "latitude": 41.8781,
        "longitude": -87.6298,
        "timezone": "America/Chicago",
        "nws_gridpoint": "LOT/76,73",
        "kalshi_high_prefix": "HIGHCHI",
        "kalshi_low_prefix": "LOWCHI",
        "settlement_station": "O'Hare"
    },
    "LA": {
        "name": "Los Angeles",
        "latitude": 34.0522,
        "longitude": -118.2437,
        "timezone": "America/Los_Angeles",
        "nws_gridpoint": "LOX/154,44",
        "kalshi_high_prefix": "HIGHLAX",
        "kalshi_low_prefix": "LOWLAX",
        "settlement_station": "LAX Airport"
    },
    "MIA": {
        "name": "Miami",
        "latitude": 25.7617,
        "longitude": -80.1918,
        "timezone": "America/New_York",
        "nws_gridpoint": "MFL/110,50",
        "kalshi_high_prefix": "HIGHMIA",
        "kalshi_low_prefix": "LOWMIA",
        "settlement_station": "Miami International Airport"
    },
    "AUS": {
        "name": "Austin",
        "latitude": 30.2672,
        "longitude": -97.7431,
        "timezone": "America/Chicago",
        "nws_gridpoint": "EWX/156,91",
        "kalshi_high_prefix": "HIGHAUS",
        "kalshi_low_prefix": "LOWAUS",
        "settlement_station": "Camp Mabry"
    },
    "DEN": {
        "name": "Denver",
        "latitude": 39.7392,
        "longitude": -104.9903,
        "timezone": "America/Denver",
        "nws_gridpoint": "BOU/62,60",
        "kalshi_high_prefix": "HIGHDEN",
        "kalshi_low_prefix": "LOWDEN",
        "settlement_station": "Denver International Airport"
    },
    "PHI": {
        "name": "Philadelphia",
        "latitude": 39.9526,
        "longitude": -75.1652,
        "timezone": "America/New_York",
        "nws_gridpoint": "PHI/49,75",
        "kalshi_high_prefix": "HIGHPHIL",
        "kalshi_low_prefix": "LOWPHIL",
        "settlement_station": "Philadelphia International Airport"
    }
}

# Active locations for trading - ALL CITIES
ACTIVE_LOCATIONS = ["NYC", "CHI", "LA", "MIA", "AUS", "DEN", "PHI"]


def get_location(location_key: str) -> dict:
    """
    Get location configuration by key.

    Args:
        location_key: Location identifier (e.g., "NYC", "CHI")

    Returns:
        Location configuration dictionary

    Raises:
        ValueError: If location_key not found
    """
    if location_key not in LOCATIONS:
        raise ValueError(f"Unknown location: {location_key}. Available: {list(LOCATIONS.keys())}")
    return LOCATIONS[location_key]


def get_active_locations() -> list:
    """Get list of active location configurations."""
    return [LOCATIONS[key] for key in ACTIVE_LOCATIONS]
