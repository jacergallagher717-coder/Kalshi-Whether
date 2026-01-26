"""
Economic Nowcast Data Collectors

Aggregates inflation/economic nowcasts from multiple sources:
1. Cleveland Fed Inflation Nowcast (daily updates)
2. Atlanta Fed GDPNow (via FRED API)
3. Manual consensus estimates

These nowcasts update more frequently than monthly BLS releases,
giving us fresher information than many market participants use.

IMPORTANT: Before each trade, manually verify nowcast values at:
- Cleveland Fed: https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting
- Atlanta Fed: https://www.atlantafed.org/cqer/research/gdpnow
"""

import requests
import os
import json
from datetime import datetime, date
from typing import Dict, Optional, List
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger("nowcast_collector")


# ============================================================
# MANUALLY UPDATED NOWCAST VALUES
# ============================================================
# Update these values before trading by checking the sources below.
# These are used when automated fetching fails (which is often).
#
# CPI/PCE Source: https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting
# GDP Source: https://www.atlantafed.org/cqer/research/gdpnow
#
# Last updated: 2026-01-26
# ============================================================

MANUAL_NOWCASTS = {
    "cpi_headline": {
        "value": 2.7,           # YoY % - check Cleveland Fed
        "period": "2025-12",    # December 2025 (last reported)
        "updated": "2026-01-26",
        "source_url": "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
    },
    "cpi_core": {
        "value": 2.6,           # YoY % - check Cleveland Fed
        "period": "2025-12",
        "updated": "2026-01-26",
        "source_url": "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
    },
    "pce_headline": {
        "value": 2.4,           # YoY % - check Cleveland Fed
        "period": "2025-12",
        "updated": "2026-01-26",
        "source_url": "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
    },
    "pce_core": {
        "value": 2.5,           # YoY % - check Cleveland Fed
        "period": "2025-12",
        "updated": "2026-01-26",
        "source_url": "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
    },
    "gdp_nowcast": {
        "value": 2.5,           # Annualized % - check Atlanta Fed GDPNow
        "period": "2026Q1",
        "updated": "2026-01-26",
        "source_url": "https://www.atlantafed.org/cqer/research/gdpnow"
    },
    # NFP consensus - update before jobs report
    "nfp_consensus": {
        "value": 50,            # Thousands of jobs expected
        "period": "2026-01",    # January 2026 report
        "updated": "2026-01-26",
        "source_url": "https://www.investing.com/economic-calendar/nonfarm-payrolls-227"
    }
}


@dataclass
class NowcastData:
    """Container for nowcast data point."""
    source: str
    indicator: str  # "cpi_headline", "cpi_core", "pce_headline", "pce_core", "gdp"
    value: float    # The nowcast value (e.g., 2.62 for 2.62% YoY)
    period: str     # e.g., "2025-12" for December 2025
    timestamp: datetime
    units: str      # "yoy_pct", "mom_pct", "annualized_pct", "thousands"

    def __repr__(self):
        return f"{self.source} {self.indicator}: {self.value:.2f} ({self.period})"


class ManualNowcastCollector:
    """
    Returns manually-updated nowcast values.

    This is the most reliable approach since automated scraping
    of Fed websites is fragile and breaks frequently.

    BEFORE EACH TRADE: Update the MANUAL_NOWCASTS dict at the top
    of this file with current values from the source URLs.
    """

    def __init__(self, cache_file: str = "./data/nowcast_cache.json"):
        self.cache_file = cache_file
        self._load_cache()

    def _load_cache(self):
        """Load cached values if they exist and are fresher than hardcoded."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cached = json.load(f)
                    # Merge cached values (they override hardcoded if newer)
                    for key, val in cached.items():
                        if key in MANUAL_NOWCASTS:
                            cached_date = val.get('updated', '2000-01-01')
                            hardcoded_date = MANUAL_NOWCASTS[key].get('updated', '2000-01-01')
                            if cached_date > hardcoded_date:
                                MANUAL_NOWCASTS[key] = val
            except Exception as e:
                logger.warning(f"Could not load nowcast cache: {e}")

    def update_nowcast(self, indicator: str, value: float, period: str = None):
        """
        Update a nowcast value (saves to cache file).

        Usage:
            collector = ManualNowcastCollector()
            collector.update_nowcast("cpi_headline", 2.75, "2026-01")
        """
        if indicator not in MANUAL_NOWCASTS:
            MANUAL_NOWCASTS[indicator] = {}

        today = date.today()
        MANUAL_NOWCASTS[indicator]['value'] = value
        MANUAL_NOWCASTS[indicator]['period'] = period or f"{today.year}-{today.month:02d}"
        MANUAL_NOWCASTS[indicator]['updated'] = str(today)

        # Save to cache
        self._save_cache()
        logger.info(f"Updated {indicator} to {value} for period {period}")

    def _save_cache(self):
        """Save current values to cache file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(MANUAL_NOWCASTS, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save nowcast cache: {e}")

    def get_nowcast(self, indicator: str) -> Optional[NowcastData]:
        """Get a specific nowcast value."""
        if indicator not in MANUAL_NOWCASTS:
            return None

        data = MANUAL_NOWCASTS[indicator]

        # Determine units based on indicator type
        if 'gdp' in indicator.lower():
            units = "annualized_pct"
        elif 'nfp' in indicator.lower() or 'jobs' in indicator.lower():
            units = "thousands"
        else:
            units = "yoy_pct"

        return NowcastData(
            source="manual",
            indicator=indicator,
            value=data['value'],
            period=data['period'],
            timestamp=datetime.utcnow(),
            units=units
        )

    def get_all_nowcasts(self) -> List[NowcastData]:
        """Get all available nowcasts."""
        nowcasts = []
        for indicator in MANUAL_NOWCASTS:
            nc = self.get_nowcast(indicator)
            if nc:
                nowcasts.append(nc)
        return nowcasts

    def check_freshness(self) -> Dict[str, bool]:
        """Check if nowcasts are fresh (updated within last 3 days)."""
        today = date.today()
        freshness = {}

        for indicator, data in MANUAL_NOWCASTS.items():
            updated_str = data.get('updated', '2000-01-01')
            try:
                updated = datetime.strptime(updated_str, '%Y-%m-%d').date()
                days_old = (today - updated).days
                freshness[indicator] = days_old <= 3
            except:
                freshness[indicator] = False

        return freshness


class AtlantaFedGDPNow:
    """
    Fetches GDP nowcast from Atlanta Fed via FRED API.

    This one actually works with the FRED API!
    Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
    """

    FRED_SERIES = "GDPNOW"
    FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.session = requests.Session()

    def get_latest_nowcast(self) -> Optional[NowcastData]:
        """Fetch latest GDPNow from FRED API."""
        if not self.api_key:
            logger.debug("No FRED API key - using manual GDP nowcast")
            return None

        try:
            params = {
                'series_id': self.FRED_SERIES,
                'api_key': self.api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 1
            }

            response = self.session.get(self.FRED_URL, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if data.get('observations'):
                    obs = data['observations'][0]
                    value = float(obs['value'])

                    logger.info(f"Fetched GDPNow from FRED: {value}%")

                    return NowcastData(
                        source="atlanta_fed_fred",
                        indicator="gdp_nowcast",
                        value=value,
                        period=obs['date'],
                        timestamp=datetime.utcnow(),
                        units="annualized_pct"
                    )
        except Exception as e:
            logger.error(f"FRED API error: {e}")

        return None


class NowcastAggregator:
    """
    Main class that aggregates all nowcast sources.

    Primary source: Manual values (most reliable)
    Secondary source: FRED API for GDP (if API key configured)
    """

    def __init__(self, fred_api_key: str = None):
        self.manual = ManualNowcastCollector()
        self.atlanta = AtlantaFedGDPNow(fred_api_key)

    def get_all_nowcasts(self) -> Dict[str, List[NowcastData]]:
        """Get all nowcasts, grouped by indicator."""
        all_nowcasts = {}

        # Get manual nowcasts
        for nc in self.manual.get_all_nowcasts():
            if nc.indicator not in all_nowcasts:
                all_nowcasts[nc.indicator] = []
            all_nowcasts[nc.indicator].append(nc)

        # Try to get live GDP from FRED (overrides manual if successful)
        gdp = self.atlanta.get_latest_nowcast()
        if gdp:
            all_nowcasts['gdp_nowcast'] = [gdp]

        return all_nowcasts

    def get_aggregate_nowcast(self, indicator: str) -> Optional[Dict]:
        """Get nowcast for a specific indicator."""
        all_nowcasts = self.get_all_nowcasts()

        if indicator not in all_nowcasts or not all_nowcasts[indicator]:
            return None

        nowcasts = all_nowcasts[indicator]

        # For now, just use the first (and usually only) value
        nc = nowcasts[0]

        # Check freshness
        freshness = self.manual.check_freshness()
        is_fresh = freshness.get(indicator, False)

        return {
            'indicator': indicator,
            'aggregate_value': nc.value,
            'spread': 0,  # Only one source
            'num_sources': 1,
            'sources': {nc.source: nc.value},
            'confidence': 0.8 if is_fresh else 0.5,
            'is_fresh': is_fresh,
            'period': nc.period,
            'timestamp': datetime.utcnow()
        }

    def print_status(self):
        """Print current nowcast status (for daily scan)."""
        print("\n=== CURRENT NOWCASTS ===")
        print("(Update these before trading at the source URLs)")
        print()

        freshness = self.manual.check_freshness()

        for indicator, data in MANUAL_NOWCASTS.items():
            fresh = "✓ FRESH" if freshness.get(indicator) else "⚠️ STALE"
            print(f"  {indicator}:")
            print(f"    Value: {data['value']}")
            print(f"    Period: {data['period']}")
            print(f"    Updated: {data['updated']} [{fresh}]")
            print(f"    Source: {data['source_url']}")
            print()


# Convenience functions
def get_cpi_nowcast() -> Optional[Dict]:
    """Quick function to get current CPI nowcast."""
    agg = NowcastAggregator()
    return agg.get_aggregate_nowcast('cpi_headline')


def update_nowcast(indicator: str, value: float, period: str = None):
    """Quick function to update a nowcast value."""
    collector = ManualNowcastCollector()
    collector.update_nowcast(indicator, value, period)


if __name__ == "__main__":
    print("=" * 60)
    print("  NOWCAST STATUS CHECK")
    print("=" * 60)

    agg = NowcastAggregator()
    agg.print_status()

    print("=" * 60)
    print("  TO UPDATE VALUES:")
    print("=" * 60)
    print()
    print("  Option 1: Edit MANUAL_NOWCASTS in this file directly")
    print()
    print("  Option 2: Use Python:")
    print("    from data.collectors.nowcast_collector import update_nowcast")
    print("    update_nowcast('cpi_headline', 2.75, '2026-01')")
    print()
    print("  Before each trade, check:")
    print("  - CPI/PCE: https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting")
    print("  - GDP: https://www.atlantafed.org/cqer/research/gdpnow")
    print("  - NFP consensus: https://www.investing.com/economic-calendar/")
    print()
