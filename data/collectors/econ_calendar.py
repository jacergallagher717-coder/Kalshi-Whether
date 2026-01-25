"""
Economic Calendar System

Tracks scheduled economic data releases that can move Kalshi markets:
- CPI (monthly, ~15th)
- PCE (monthly, end of month)
- Jobs Report/NFP (first Friday)
- Fed Decisions (FOMC schedule)
- GDP (quarterly)

The calendar tells us WHEN to be ready to trade, and helps us
pre-position before information drops.
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json
import os

from utils.logger import get_logger

logger = get_logger("econ_calendar")


class EventType(Enum):
    """Types of economic events."""
    CPI = "cpi"              # Consumer Price Index
    CORE_CPI = "core_cpi"    # Core CPI (ex food/energy)
    PCE = "pce"              # Personal Consumption Expenditure
    CORE_PCE = "core_pce"    # Core PCE (Fed's preferred measure)
    NFP = "nfp"              # Non-Farm Payrolls
    UNEMPLOYMENT = "unemployment"
    FED_DECISION = "fed_decision"
    FED_MINUTES = "fed_minutes"
    GDP = "gdp"
    RETAIL_SALES = "retail_sales"
    ISM_MFG = "ism_manufacturing"
    ISM_SVC = "ism_services"


class EventImpact(Enum):
    """Expected market impact level."""
    HIGH = "high"       # Major market mover (CPI, NFP, Fed)
    MEDIUM = "medium"   # Significant (PCE, GDP)
    LOW = "low"         # Minor impact


@dataclass
class EconEvent:
    """Represents a scheduled economic event."""
    event_type: EventType
    release_date: date
    release_time: str  # "08:30 ET" format
    period: str        # "2025-12" or "2025Q4"
    impact: EventImpact
    description: str

    # Optional: consensus forecast (updated as event approaches)
    consensus: Optional[float] = None
    prior: Optional[float] = None

    # For tracking
    actual: Optional[float] = None  # Filled after release
    surprise: Optional[float] = None  # actual - consensus

    @property
    def is_upcoming(self) -> bool:
        """Check if event is in the future."""
        return self.release_date >= date.today()

    @property
    def hours_until(self) -> float:
        """Hours until event release."""
        now = datetime.now()
        # Parse release time
        hour, minute = 8, 30  # Default
        if ":" in self.release_time:
            parts = self.release_time.replace(" ET", "").split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0

        release_dt = datetime.combine(self.release_date, datetime.min.time())
        release_dt = release_dt.replace(hour=hour, minute=minute)

        delta = release_dt - now
        return delta.total_seconds() / 3600

    def __repr__(self):
        status = "RELEASED" if self.actual is not None else f"{self.hours_until:.1f}h away"
        return f"{self.event_type.value.upper()} ({self.period}) - {self.release_date} [{status}]"


class EconCalendar:
    """
    Manages the economic calendar and identifies trading windows.

    Key features:
    1. Pre-loaded with 2026 release schedule
    2. Tracks upcoming events
    3. Identifies optimal trading windows (24-48h before release)
    4. Stores historical actuals for backtesting
    """

    def __init__(self, data_dir: str = "./data/calendar"):
        self.data_dir = data_dir
        self.events: List[EconEvent] = []

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Load the calendar
        self._load_2026_calendar()

    def _load_2026_calendar(self):
        """Load the 2026 economic calendar."""
        # CPI releases (typically around 15th of month)
        # These are for PRIOR month data
        cpi_dates = [
            ("2026-01-15", "2025-12"),  # December 2025 CPI
            ("2026-02-12", "2026-01"),  # January 2026 CPI
            ("2026-03-12", "2026-02"),
            ("2026-04-10", "2026-03"),
            ("2026-05-13", "2026-04"),
            ("2026-06-11", "2026-05"),
            ("2026-07-14", "2026-06"),
            ("2026-08-12", "2026-07"),
            ("2026-09-11", "2026-08"),
            ("2026-10-13", "2026-09"),
            ("2026-11-12", "2026-10"),
            ("2026-12-10", "2026-11"),
        ]

        for release_str, period in cpi_dates:
            release = datetime.strptime(release_str, "%Y-%m-%d").date()
            self.events.append(EconEvent(
                event_type=EventType.CPI,
                release_date=release,
                release_time="08:30 ET",
                period=period,
                impact=EventImpact.HIGH,
                description=f"CPI for {period}"
            ))

        # Jobs Report (first Friday of month)
        nfp_dates = [
            ("2026-01-10", "2025-12"),
            ("2026-02-06", "2026-01"),
            ("2026-03-06", "2026-02"),
            ("2026-04-03", "2026-03"),
            ("2026-05-08", "2026-04"),
            ("2026-06-05", "2026-05"),
            ("2026-07-02", "2026-06"),
            ("2026-08-07", "2026-07"),
            ("2026-09-04", "2026-08"),
            ("2026-10-02", "2026-09"),
            ("2026-11-06", "2026-10"),
            ("2026-12-04", "2026-11"),
        ]

        for release_str, period in nfp_dates:
            release = datetime.strptime(release_str, "%Y-%m-%d").date()
            self.events.append(EconEvent(
                event_type=EventType.NFP,
                release_date=release,
                release_time="08:30 ET",
                period=period,
                impact=EventImpact.HIGH,
                description=f"Non-Farm Payrolls for {period}"
            ))

        # Fed Decisions (FOMC meetings)
        fomc_dates = [
            ("2026-01-29", "2026-01"),
            ("2026-03-18", "2026-03"),
            ("2026-05-06", "2026-05"),
            ("2026-06-17", "2026-06"),
            ("2026-07-29", "2026-07"),
            ("2026-09-16", "2026-09"),
            ("2026-11-04", "2026-11"),
            ("2026-12-16", "2026-12"),
        ]

        for release_str, period in fomc_dates:
            release = datetime.strptime(release_str, "%Y-%m-%d").date()
            self.events.append(EconEvent(
                event_type=EventType.FED_DECISION,
                release_date=release,
                release_time="14:00 ET",
                period=period,
                impact=EventImpact.HIGH,
                description=f"FOMC Rate Decision {period}"
            ))

        # PCE releases (end of month, ~2 weeks after CPI)
        pce_dates = [
            ("2026-01-31", "2025-12"),
            ("2026-02-27", "2026-01"),
            ("2026-03-27", "2026-02"),
            ("2026-04-30", "2026-03"),
            ("2026-05-29", "2026-04"),
            ("2026-06-26", "2026-05"),
            ("2026-07-31", "2026-06"),
            ("2026-08-28", "2026-07"),
            ("2026-09-25", "2026-08"),
            ("2026-10-30", "2026-09"),
            ("2026-11-25", "2026-10"),
            ("2026-12-23", "2026-11"),
        ]

        for release_str, period in pce_dates:
            release = datetime.strptime(release_str, "%Y-%m-%d").date()
            self.events.append(EconEvent(
                event_type=EventType.PCE,
                release_date=release,
                release_time="08:30 ET",
                period=period,
                impact=EventImpact.MEDIUM,
                description=f"PCE Inflation for {period}"
            ))

        # Sort by date
        self.events.sort(key=lambda e: e.release_date)
        logger.info(f"Loaded {len(self.events)} economic events for 2026")

    def get_upcoming_events(self, days: int = 7) -> List[EconEvent]:
        """Get events in the next N days."""
        today = date.today()
        cutoff = today + timedelta(days=days)

        return [
            e for e in self.events
            if today <= e.release_date <= cutoff
        ]

    def get_next_event(self, event_type: EventType = None) -> Optional[EconEvent]:
        """Get the next upcoming event, optionally filtered by type."""
        today = date.today()

        for event in self.events:
            if event.release_date >= today:
                if event_type is None or event.event_type == event_type:
                    return event

        return None

    def get_trading_windows(self) -> List[Dict]:
        """
        Identify current trading windows.

        A trading window is open when:
        - We're 24-48 hours before a HIGH impact release
        - We have fresh nowcast data
        - We haven't already positioned

        Returns list of tradeable opportunities.
        """
        windows = []
        today = date.today()

        for event in self.events:
            if event.impact != EventImpact.HIGH:
                continue

            hours = event.hours_until

            # Prime window: 24-48 hours before release
            if 24 <= hours <= 48:
                windows.append({
                    'event': event,
                    'window_type': 'prime',
                    'hours_until': hours,
                    'recommendation': 'Position now if nowcast diverges from market'
                })

            # Late window: 6-24 hours before release
            elif 6 <= hours < 24:
                windows.append({
                    'event': event,
                    'window_type': 'late',
                    'hours_until': hours,
                    'recommendation': 'Last chance to position, check spreads'
                })

            # Imminent: <6 hours
            elif 0 < hours < 6:
                windows.append({
                    'event': event,
                    'window_type': 'imminent',
                    'hours_until': hours,
                    'recommendation': 'Data release imminent, prepare to exit or hold'
                })

        return windows

    def update_actual(self, event_type: EventType, period: str, actual: float, consensus: float = None):
        """Update an event with actual released value."""
        for event in self.events:
            if event.event_type == event_type and event.period == period:
                event.actual = actual
                if consensus:
                    event.consensus = consensus
                    event.surprise = actual - consensus
                logger.info(f"Updated {event_type.value} {period}: actual={actual}, surprise={event.surprise}")
                return

        logger.warning(f"Event not found: {event_type.value} {period}")

    def to_dict(self) -> List[Dict]:
        """Export calendar as list of dicts."""
        return [
            {
                'event_type': e.event_type.value,
                'release_date': str(e.release_date),
                'release_time': e.release_time,
                'period': e.period,
                'impact': e.impact.value,
                'description': e.description,
                'consensus': e.consensus,
                'actual': e.actual,
                'surprise': e.surprise
            }
            for e in self.events
        ]

    def save(self, filepath: str = None):
        """Save calendar to JSON."""
        filepath = filepath or os.path.join(self.data_dir, "calendar.json")
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved calendar to {filepath}")


def get_next_cpi_window() -> Optional[Dict]:
    """Quick function to check if we're in a CPI trading window."""
    cal = EconCalendar()
    next_cpi = cal.get_next_event(EventType.CPI)

    if not next_cpi:
        return None

    hours = next_cpi.hours_until

    return {
        'event': str(next_cpi),
        'release_date': str(next_cpi.release_date),
        'hours_until': hours,
        'in_prime_window': 24 <= hours <= 48,
        'in_late_window': 6 <= hours < 24,
        'recommendation': 'Position now' if 24 <= hours <= 48 else (
            'Last chance' if 6 <= hours < 24 else 'Wait for next window'
        )
    }


if __name__ == "__main__":
    print("Testing Economic Calendar...")

    cal = EconCalendar()

    print("\nUpcoming events (next 30 days):")
    for event in cal.get_upcoming_events(days=30):
        print(f"  {event}")

    print("\nNext CPI:")
    cpi = cal.get_next_event(EventType.CPI)
    if cpi:
        print(f"  {cpi}")
        print(f"  Hours until: {cpi.hours_until:.1f}")

    print("\nCurrent trading windows:")
    for window in cal.get_trading_windows():
        print(f"  {window['event']}")
        print(f"    Window: {window['window_type']}")
        print(f"    Hours: {window['hours_until']:.1f}")
        print(f"    Recommendation: {window['recommendation']}")

    print("\nNext CPI Window Check:")
    window = get_next_cpi_window()
    if window:
        print(f"  {window}")
