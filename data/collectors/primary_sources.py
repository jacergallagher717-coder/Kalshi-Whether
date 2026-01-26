"""
Primary Source Data Pipeline

Principle: Sit one layer closer to the source than the crowd.
Most traders react to headlines. We react to primary data.

This module provides direct access to:
1. BLS release schedules (CPI, NFP, PPI)
2. Federal Reserve calendars (FOMC, speeches)
3. Cleveland Fed Nowcast (daily updates)
4. Court dockets (PACER integration placeholder)
"""

import requests
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import json
import os
import hashlib

from utils.logger import get_logger

logger = get_logger("primary_sources")


@dataclass
class DataRelease:
    """A scheduled data release from a primary source."""
    source: str           # "BLS", "FED", "CLEVELAND_FED", etc.
    release_type: str     # "CPI", "NFP", "FOMC", etc.
    release_date: date
    release_time: str     # "08:30 ET"
    url: str              # Direct link to source
    last_checked: datetime
    data_hash: str = ""   # Hash to detect changes

    def __repr__(self):
        return f"{self.source}/{self.release_type} - {self.release_date} {self.release_time}"


class BLSCalendar:
    """
    Direct access to Bureau of Labor Statistics release schedule.

    Primary source: https://www.bls.gov/schedule/news_release/

    BLS publishes their release schedule months in advance.
    This is the canonical source - not news articles about it.
    """

    SCHEDULE_URL = "https://www.bls.gov/schedule/news_release/"
    CPI_URL = "https://www.bls.gov/cpi/"
    EMPLOYMENT_URL = "https://www.bls.gov/ces/"

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (research bot)'
        })

    def get_upcoming_releases(self, days: int = 30) -> List[DataRelease]:
        """
        Fetch upcoming BLS releases directly from their schedule.

        This is faster than waiting for news to report it.
        """
        releases = []

        try:
            response = self.session.get(self.SCHEDULE_URL, timeout=30)
            if response.status_code == 200:
                releases = self._parse_schedule(response.text, days)
                self._cache_schedule(releases)
        except Exception as e:
            logger.error(f"BLS schedule fetch failed: {e}")
            releases = self._load_cached_schedule()

        return releases

    def _parse_schedule(self, html: str, days: int) -> List[DataRelease]:
        """Parse the BLS schedule page."""
        releases = []
        soup = BeautifulSoup(html, 'html.parser')
        today = date.today()
        cutoff = today + timedelta(days=days)

        # BLS schedule is in a table format
        # Look for rows with dates and release names
        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    # Try to parse date and release type
                    date_text = cells[0].get_text(strip=True)
                    release_text = cells[1].get_text(strip=True)

                    try:
                        # Parse various date formats BLS might use
                        release_date = self._parse_date(date_text)
                        if release_date and today <= release_date <= cutoff:
                            release_type = self._categorize_release(release_text)
                            if release_type:
                                releases.append(DataRelease(
                                    source="BLS",
                                    release_type=release_type,
                                    release_date=release_date,
                                    release_time="08:30 ET",
                                    url=self.SCHEDULE_URL,
                                    last_checked=datetime.utcnow()
                                ))
                    except:
                        continue

        return releases

    def _parse_date(self, text: str) -> Optional[date]:
        """Try to parse date from various formats."""
        import re

        # Common patterns: "January 15, 2026", "01/15/2026", "Jan 15"
        patterns = [
            r'(\w+)\s+(\d{1,2}),?\s*(\d{4})',  # January 15, 2026
            r'(\d{1,2})/(\d{1,2})/(\d{4})',     # 01/15/2026
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        if groups[0].isalpha():
                            # Month name format
                            return datetime.strptime(f"{groups[0]} {groups[1]} {groups[2]}", "%B %d %Y").date()
                        else:
                            # Numeric format
                            return datetime.strptime(f"{groups[0]}/{groups[1]}/{groups[2]}", "%m/%d/%Y").date()
                except:
                    continue

        return None

    def _categorize_release(self, text: str) -> Optional[str]:
        """Categorize the release type."""
        text_lower = text.lower()

        if 'consumer price' in text_lower or 'cpi' in text_lower:
            return 'CPI'
        elif 'employment situation' in text_lower or 'payroll' in text_lower:
            return 'NFP'
        elif 'producer price' in text_lower or 'ppi' in text_lower:
            return 'PPI'
        elif 'jobless claims' in text_lower:
            return 'CLAIMS'

        return None

    def _cache_schedule(self, releases: List[DataRelease]):
        """Cache the schedule locally."""
        cache_file = os.path.join(self.cache_dir, "bls_schedule.json")
        data = [
            {
                'source': r.source,
                'release_type': r.release_type,
                'release_date': str(r.release_date),
                'release_time': r.release_time,
                'url': r.url
            }
            for r in releases
        ]
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_cached_schedule(self) -> List[DataRelease]:
        """Load from cache if fetch fails."""
        cache_file = os.path.join(self.cache_dir, "bls_schedule.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return [
                    DataRelease(
                        source=d['source'],
                        release_type=d['release_type'],
                        release_date=datetime.strptime(d['release_date'], '%Y-%m-%d').date(),
                        release_time=d['release_time'],
                        url=d['url'],
                        last_checked=datetime.utcnow()
                    )
                    for d in data
                ]
            except:
                pass
        return []


class FedCalendar:
    """
    Direct access to Federal Reserve calendars.

    Primary sources:
    - FOMC Calendar: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    - Fed Speeches: https://www.federalreserve.gov/newsevents/speeches.htm

    The Fed publishes the full FOMC calendar for the year.
    """

    FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    SPEECHES_URL = "https://www.federalreserve.gov/newsevents/speeches.htm"

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()

    def get_fomc_schedule(self, year: int = 2026) -> List[DataRelease]:
        """
        Fetch FOMC meeting schedule directly from the Fed.

        This is the authoritative source - press releases cite this.
        """
        releases = []

        try:
            response = self.session.get(self.FOMC_URL, timeout=30)
            if response.status_code == 200:
                releases = self._parse_fomc_calendar(response.text, year)
        except Exception as e:
            logger.error(f"Fed calendar fetch failed: {e}")

        return releases

    def _parse_fomc_calendar(self, html: str, year: int) -> List[DataRelease]:
        """Parse the FOMC calendar page."""
        releases = []
        soup = BeautifulSoup(html, 'html.parser')

        # The Fed calendar has a specific structure
        # Look for meeting dates
        import re

        # Find year section
        year_pattern = str(year)
        text = soup.get_text()

        # FOMC dates are typically listed as "January 28-29" or "March 18-19"
        date_pattern = r'(\w+)\s+(\d{1,2})(?:-(\d{1,2}))?(?:,?\s*' + year_pattern + r')?'

        # Known 2026 FOMC dates (backup if parsing fails)
        known_2026_dates = [
            ("January", 28, 29),
            ("March", 17, 18),
            ("May", 5, 6),
            ("June", 16, 17),
            ("July", 28, 29),
            ("September", 15, 16),
            ("November", 3, 4),
            ("December", 15, 16),
        ]

        for month, day1, day2 in known_2026_dates:
            try:
                meeting_date = datetime.strptime(f"{month} {day2} {year}", "%B %d %Y").date()
                releases.append(DataRelease(
                    source="FED",
                    release_type="FOMC",
                    release_date=meeting_date,
                    release_time="14:00 ET",
                    url=self.FOMC_URL,
                    last_checked=datetime.utcnow()
                ))
            except:
                continue

        return releases


class ChangeDetector:
    """
    Monitor primary sources for changes.

    Principle: Be notified of updates before the crowd notices.

    This hashes page content and alerts when it changes.
    """

    def __init__(self, storage_dir: str = "./data/hashes"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.session = requests.Session()

    def check_for_changes(self, url: str, name: str) -> Dict:
        """
        Check if a URL's content has changed since last check.

        Returns dict with:
        - changed: bool
        - old_hash: str
        - new_hash: str
        - checked_at: datetime
        """
        hash_file = os.path.join(self.storage_dir, f"{name}.hash")

        try:
            response = self.session.get(url, timeout=30)
            new_hash = hashlib.md5(response.text.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return {'changed': False, 'error': str(e)}

        old_hash = None
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                old_hash = f.read().strip()

        # Update stored hash
        with open(hash_file, 'w') as f:
            f.write(new_hash)

        changed = old_hash is not None and old_hash != new_hash

        if changed:
            logger.info(f"CHANGE DETECTED: {name} at {url}")

        return {
            'changed': changed,
            'old_hash': old_hash,
            'new_hash': new_hash,
            'checked_at': datetime.utcnow().isoformat(),
            'url': url
        }

    def monitor_sources(self, sources: List[Dict]) -> List[Dict]:
        """
        Check multiple sources for changes.

        Args:
            sources: List of {'name': str, 'url': str}

        Returns:
            List of sources that changed
        """
        changes = []

        for source in sources:
            result = self.check_for_changes(source['url'], source['name'])
            if result.get('changed'):
                changes.append({
                    'name': source['name'],
                    **result
                })

        return changes


class PrimarySourceAggregator:
    """
    Main class that aggregates all primary sources.

    Usage:
        agg = PrimarySourceAggregator()

        # Get upcoming releases
        releases = agg.get_all_upcoming_releases()

        # Check for source changes
        changes = agg.check_for_updates()
    """

    # Sources to monitor for changes
    MONITORED_SOURCES = [
        {'name': 'cleveland_fed_nowcast', 'url': 'https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting'},
        {'name': 'bls_schedule', 'url': 'https://www.bls.gov/schedule/news_release/'},
        {'name': 'fomc_calendar', 'url': 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'},
        {'name': 'fed_speeches', 'url': 'https://www.federalreserve.gov/newsevents/speeches.htm'},
    ]

    def __init__(self):
        self.bls = BLSCalendar()
        self.fed = FedCalendar()
        self.detector = ChangeDetector()

    def get_all_upcoming_releases(self, days: int = 30) -> List[DataRelease]:
        """Get all upcoming releases from all sources."""
        releases = []

        releases.extend(self.bls.get_upcoming_releases(days))
        releases.extend(self.fed.get_fomc_schedule())

        # Sort by date
        releases.sort(key=lambda r: r.release_date)

        # Filter to upcoming only
        today = date.today()
        cutoff = today + timedelta(days=days)
        releases = [r for r in releases if today <= r.release_date <= cutoff]

        return releases

    def check_for_updates(self) -> List[Dict]:
        """
        Check all monitored sources for updates.

        Run this periodically (e.g., every hour) to catch changes early.
        """
        return self.detector.monitor_sources(self.MONITORED_SOURCES)

    def get_next_release(self, release_type: str = None) -> Optional[DataRelease]:
        """Get the next upcoming release, optionally filtered by type."""
        releases = self.get_all_upcoming_releases(days=60)

        for release in releases:
            if release_type is None or release.release_type == release_type:
                return release

        return None


if __name__ == "__main__":
    print("Testing Primary Source Pipeline...")

    agg = PrimarySourceAggregator()

    print("\nUpcoming releases (next 30 days):")
    for release in agg.get_all_upcoming_releases(30):
        print(f"  {release}")

    print("\nNext CPI release:")
    cpi = agg.get_next_release("CPI")
    if cpi:
        print(f"  {cpi}")

    print("\nNext FOMC:")
    fomc = agg.get_next_release("FOMC")
    if fomc:
        print(f"  {fomc}")

    print("\nChecking for source changes...")
    changes = agg.check_for_updates()
    if changes:
        print("  CHANGES DETECTED:")
        for c in changes:
            print(f"    {c['name']}")
    else:
        print("  No changes detected")
