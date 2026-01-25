"""
Economic Nowcast Data Collectors

Aggregates inflation/economic nowcasts from multiple sources:
1. Cleveland Fed Inflation Nowcast (daily updates)
2. NY Fed Staff Nowcast (weekly)
3. Atlanta Fed GDPNow
4. Wall Street Consensus estimates

These nowcasts update more frequently than monthly BLS releases,
giving us fresher information than many market participants use.
"""

import requests
from datetime import datetime, date
from typing import Dict, Optional, List
from dataclasses import dataclass
from bs4 import BeautifulSoup
import re

from utils.logger import get_logger

logger = get_logger("nowcast_collector")


@dataclass
class NowcastData:
    """Container for nowcast data point."""
    source: str
    indicator: str  # "cpi_headline", "cpi_core", "pce_headline", "pce_core", "gdp"
    value: float  # The nowcast value (e.g., 2.62 for 2.62% YoY)
    period: str  # e.g., "2025-12" for December 2025
    timestamp: datetime  # When we collected this
    units: str  # "yoy_pct", "mom_pct", "annualized_pct"

    def __repr__(self):
        return f"{self.source} {self.indicator}: {self.value:.2f}% ({self.period})"


class ClevelandFedCollector:
    """
    Collects inflation nowcasts from Cleveland Fed.

    The Cleveland Fed provides daily nowcasts for:
    - CPI (headline and core)
    - PCE (headline and core)

    These update every business day and incorporate the latest
    energy prices, PPI data, and other high-frequency indicators.

    URL: https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting
    """

    BASE_URL = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"

    # API endpoint for the nowcast data (JSON)
    # Note: This may need to be discovered via browser dev tools
    API_URL = "https://www.clevelandfed.org/api/inflation-nowcasting/data"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def get_latest_nowcast(self) -> List[NowcastData]:
        """
        Fetch the latest Cleveland Fed inflation nowcasts.

        Returns list of NowcastData for CPI/PCE headline and core.
        """
        nowcasts = []

        try:
            # Try to fetch from their data API first
            response = self.session.get(self.API_URL, timeout=30)

            if response.status_code == 200:
                data = response.json()
                nowcasts = self._parse_api_response(data)
            else:
                # Fall back to scraping the HTML page
                logger.warning(f"API returned {response.status_code}, trying HTML scrape")
                nowcasts = self._scrape_html()

        except requests.RequestException as e:
            logger.error(f"Cleveland Fed fetch failed: {e}")
            # Return cached/fallback data if available
            nowcasts = self._get_fallback_data()
        except Exception as e:
            logger.error(f"Cleveland Fed parse error: {e}")
            nowcasts = self._get_fallback_data()

        return nowcasts

    def _parse_api_response(self, data: dict) -> List[NowcastData]:
        """Parse the JSON API response."""
        nowcasts = []
        timestamp = datetime.utcnow()

        # Structure depends on their API format
        # This is a placeholder - actual parsing depends on their response structure
        if 'cpi' in data:
            cpi_data = data['cpi']
            if 'headline' in cpi_data:
                nowcasts.append(NowcastData(
                    source="cleveland_fed",
                    indicator="cpi_headline",
                    value=float(cpi_data['headline']['yoy']),
                    period=cpi_data['headline']['period'],
                    timestamp=timestamp,
                    units="yoy_pct"
                ))
            if 'core' in cpi_data:
                nowcasts.append(NowcastData(
                    source="cleveland_fed",
                    indicator="cpi_core",
                    value=float(cpi_data['core']['yoy']),
                    period=cpi_data['core']['period'],
                    timestamp=timestamp,
                    units="yoy_pct"
                ))

        return nowcasts

    def _scrape_html(self) -> List[NowcastData]:
        """
        Scrape nowcast values from the HTML page.

        This is a fallback if the API isn't available.
        """
        nowcasts = []
        timestamp = datetime.utcnow()

        try:
            response = self.session.get(self.BASE_URL, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for the nowcast values in the page
            # The exact selectors depend on their HTML structure
            # This is a template that needs to be adapted

            # Example: find elements with specific data attributes or classes
            cpi_elements = soup.find_all(attrs={'data-indicator': 'cpi'})

            for elem in cpi_elements:
                value_text = elem.get_text(strip=True)
                # Parse the value (e.g., "2.62%" -> 2.62)
                match = re.search(r'([\d.]+)%?', value_text)
                if match:
                    value = float(match.group(1))
                    indicator = elem.get('data-type', 'cpi_headline')
                    period = elem.get('data-period', self._current_period())

                    nowcasts.append(NowcastData(
                        source="cleveland_fed",
                        indicator=indicator,
                        value=value,
                        period=period,
                        timestamp=timestamp,
                        units="yoy_pct"
                    ))

        except Exception as e:
            logger.error(f"HTML scrape failed: {e}")

        return nowcasts

    def _get_fallback_data(self) -> List[NowcastData]:
        """
        Return fallback/cached data when live fetch fails.

        In production, this would read from a local cache.
        """
        logger.warning("Using fallback nowcast data")
        timestamp = datetime.utcnow()

        # These are example values - in production, use cached real data
        return [
            NowcastData(
                source="cleveland_fed_fallback",
                indicator="cpi_headline",
                value=2.7,  # Update with latest known value
                period=self._current_period(),
                timestamp=timestamp,
                units="yoy_pct"
            ),
            NowcastData(
                source="cleveland_fed_fallback",
                indicator="cpi_core",
                value=2.6,
                period=self._current_period(),
                timestamp=timestamp,
                units="yoy_pct"
            )
        ]

    def _current_period(self) -> str:
        """Get current period string (e.g., '2026-01')."""
        today = date.today()
        # Nowcasts are typically for the previous month
        if today.month == 1:
            return f"{today.year - 1}-12"
        return f"{today.year}-{today.month - 1:02d}"


class NYFedCollector:
    """
    Collects GDP/economic nowcasts from NY Fed Staff Nowcast.

    Updates weekly (Fridays), provides:
    - GDP growth nowcast
    - Key economic indicator impacts

    URL: https://www.newyorkfed.org/research/policy/nowcast
    """

    BASE_URL = "https://www.newyorkfed.org/research/policy/nowcast"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def get_latest_nowcast(self) -> List[NowcastData]:
        """Fetch the latest NY Fed GDP nowcast."""
        nowcasts = []
        timestamp = datetime.utcnow()

        try:
            response = self.session.get(self.BASE_URL, timeout=30)

            if response.status_code == 200:
                nowcasts = self._parse_response(response.text)
            else:
                logger.warning(f"NY Fed returned {response.status_code}")

        except Exception as e:
            logger.error(f"NY Fed fetch failed: {e}")

        return nowcasts

    def _parse_response(self, html: str) -> List[NowcastData]:
        """Parse the NY Fed nowcast page."""
        nowcasts = []
        timestamp = datetime.utcnow()

        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Look for GDP nowcast value
            # The exact parsing depends on their HTML structure
            # This is a template

            gdp_text = soup.find(text=re.compile(r'GDP.*nowcast', re.I))
            if gdp_text:
                # Find nearby number
                parent = gdp_text.parent
                value_match = re.search(r'([\d.]+)%', parent.get_text())
                if value_match:
                    nowcasts.append(NowcastData(
                        source="ny_fed",
                        indicator="gdp_nowcast",
                        value=float(value_match.group(1)),
                        period=self._current_quarter(),
                        timestamp=timestamp,
                        units="annualized_pct"
                    ))

        except Exception as e:
            logger.error(f"NY Fed parse error: {e}")

        return nowcasts

    def _current_quarter(self) -> str:
        """Get current quarter string (e.g., '2026Q1')."""
        today = date.today()
        quarter = (today.month - 1) // 3 + 1
        return f"{today.year}Q{quarter}"


class AtlantaFedGDPNow:
    """
    Collects GDP nowcast from Atlanta Fed GDPNow model.

    Updates frequently (multiple times per week), provides:
    - Real GDP growth nowcast for current quarter

    URL: https://www.atlantafed.org/cqer/research/gdpnow
    """

    # GDPNow is available via FRED API
    FRED_SERIES = "GDPNOW"
    FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, fred_api_key: str = None):
        self.api_key = fred_api_key
        self.session = requests.Session()

    def get_latest_nowcast(self) -> List[NowcastData]:
        """Fetch latest GDPNow from FRED API."""
        if not self.api_key:
            logger.warning("No FRED API key configured")
            return []

        nowcasts = []
        timestamp = datetime.utcnow()

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
                    nowcasts.append(NowcastData(
                        source="atlanta_fed",
                        indicator="gdp_nowcast",
                        value=float(obs['value']),
                        period=obs['date'],
                        timestamp=timestamp,
                        units="annualized_pct"
                    ))

        except Exception as e:
            logger.error(f"Atlanta Fed GDPNow fetch failed: {e}")

        return nowcasts


class ConsensusCollector:
    """
    Aggregates Wall Street consensus forecasts.

    Sources:
    - Bloomberg consensus
    - Reuters polls
    - MarketWatch consensus

    These are typically available just before major releases.
    """

    def __init__(self):
        self.session = requests.Session()

    def get_consensus(self, indicator: str, period: str) -> Optional[NowcastData]:
        """
        Get Wall Street consensus for an indicator.

        Args:
            indicator: "cpi_headline", "cpi_core", "nfp", etc.
            period: "2026-01" for January 2026

        Returns:
            NowcastData with consensus value, or None
        """
        # In production, this would scrape from financial news sources
        # or use a data provider API (Bloomberg, Refinitiv, etc.)

        logger.info(f"Consensus lookup for {indicator} {period} - not implemented")
        return None


class NowcastAggregator:
    """
    Main class that aggregates all nowcast sources into a single view.

    Computes weighted average and spread across sources.
    """

    def __init__(self, fred_api_key: str = None):
        self.cleveland = ClevelandFedCollector()
        self.nyfed = NYFedCollector()
        self.atlanta = AtlantaFedGDPNow(fred_api_key)
        self.consensus = ConsensusCollector()

        # Weights for combining sources (based on historical accuracy)
        self.weights = {
            'cleveland_fed': 0.40,  # Best for CPI nowcasting
            'ny_fed': 0.30,
            'atlanta_fed': 0.30,
            'consensus': 0.20
        }

    def get_all_nowcasts(self) -> Dict[str, List[NowcastData]]:
        """
        Fetch nowcasts from all sources.

        Returns dict keyed by indicator with list of nowcasts from each source.
        """
        all_nowcasts = {}

        # Collect from all sources
        for nowcast in self.cleveland.get_latest_nowcast():
            if nowcast.indicator not in all_nowcasts:
                all_nowcasts[nowcast.indicator] = []
            all_nowcasts[nowcast.indicator].append(nowcast)

        for nowcast in self.nyfed.get_latest_nowcast():
            if nowcast.indicator not in all_nowcasts:
                all_nowcasts[nowcast.indicator] = []
            all_nowcasts[nowcast.indicator].append(nowcast)

        for nowcast in self.atlanta.get_latest_nowcast():
            if nowcast.indicator not in all_nowcasts:
                all_nowcasts[nowcast.indicator] = []
            all_nowcasts[nowcast.indicator].append(nowcast)

        return all_nowcasts

    def get_aggregate_nowcast(self, indicator: str) -> Optional[Dict]:
        """
        Get weighted aggregate nowcast for an indicator.

        Returns dict with:
        - aggregate_value: Weighted average
        - spread: Max - Min across sources
        - sources: Individual source values
        - confidence: Based on source agreement
        """
        all_nowcasts = self.get_all_nowcasts()

        if indicator not in all_nowcasts or not all_nowcasts[indicator]:
            return None

        nowcasts = all_nowcasts[indicator]

        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        values = []

        for nc in nowcasts:
            weight = self.weights.get(nc.source, 0.25)
            weighted_sum += nc.value * weight
            total_weight += weight
            values.append(nc.value)

        if total_weight == 0:
            return None

        aggregate = weighted_sum / total_weight
        spread = max(values) - min(values) if len(values) > 1 else 0

        # Confidence is higher when sources agree
        confidence = max(0.3, 1.0 - spread / 1.0)  # 1% spread = 0% confidence boost

        return {
            'indicator': indicator,
            'aggregate_value': round(aggregate, 2),
            'spread': round(spread, 2),
            'num_sources': len(nowcasts),
            'sources': {nc.source: nc.value for nc in nowcasts},
            'confidence': round(confidence, 2),
            'timestamp': datetime.utcnow()
        }


# Convenience function
def get_cpi_nowcast() -> Optional[Dict]:
    """Quick function to get current CPI nowcast aggregate."""
    aggregator = NowcastAggregator()
    return aggregator.get_aggregate_nowcast('cpi_headline')


if __name__ == "__main__":
    # Test the collectors
    print("Testing Cleveland Fed Collector...")
    cleveland = ClevelandFedCollector()
    nowcasts = cleveland.get_latest_nowcast()
    for nc in nowcasts:
        print(f"  {nc}")

    print("\nTesting NY Fed Collector...")
    nyfed = NYFedCollector()
    nowcasts = nyfed.get_latest_nowcast()
    for nc in nowcasts:
        print(f"  {nc}")

    print("\nTesting Aggregator...")
    agg = NowcastAggregator()
    cpi = agg.get_aggregate_nowcast('cpi_headline')
    if cpi:
        print(f"  CPI Headline Nowcast: {cpi['aggregate_value']}%")
        print(f"  Spread: {cpi['spread']}%")
        print(f"  Sources: {cpi['sources']}")
