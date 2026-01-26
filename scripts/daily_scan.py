#!/usr/bin/env python3
"""
Daily Market Scanner

Run this every morning to:
1. Check primary sources for updates
2. Get latest nowcasts
3. Find divergences vs Kalshi pricing
4. Generate trade candidates

Usage:
    python scripts/daily_scan.py
"""

import sys
import os
from datetime import datetime, date, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collectors.primary_sources import PrimarySourceAggregator
from data.collectors.econ_calendar import EconCalendar, EventType
from data.collectors.nowcast_collector import NowcastAggregator
from models.bayesian_updater import BayesianUpdater, Evidence
from utils.logger import setup_logger, get_logger

setup_logger()
logger = get_logger("daily_scan")


def print_header(text: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def main():
    print_header(f"DAILY MARKET SCAN - {date.today()}")

    # 1. Check for source updates
    print_header("1. PRIMARY SOURCE CHANGES")
    sources = PrimarySourceAggregator()
    changes = sources.check_for_updates()

    if changes:
        print("⚠️  CHANGES DETECTED:")
        for change in changes:
            print(f"   - {change['name']}: content updated")
    else:
        print("   No changes detected in monitored sources")

    # 2. Upcoming releases
    print_header("2. UPCOMING RELEASES (Next 14 days)")
    calendar = EconCalendar()
    upcoming = calendar.get_upcoming_events(days=14)

    if upcoming:
        for event in upcoming:
            hours = event.hours_until
            if hours < 48:
                flag = "🔥 IMMINENT" if hours < 24 else "⏰ SOON"
            else:
                flag = ""
            print(f"   {event.release_date} | {event.event_type.value.upper():10} | {hours:.0f}h away {flag}")
    else:
        print("   No releases in next 14 days")

    # 3. Current trading windows
    print_header("3. ACTIVE TRADING WINDOWS")
    windows = calendar.get_trading_windows()

    if windows:
        for w in windows:
            print(f"   {w['event']}")
            print(f"      Window: {w['window_type']}")
            print(f"      Action: {w['recommendation']}")
    else:
        print("   No active trading windows")

    # 4. Nowcast data
    print_header("4. CURRENT NOWCASTS")
    nowcast_agg = NowcastAggregator()

    for indicator in ['cpi_headline', 'cpi_core', 'gdp_nowcast']:
        nowcast = nowcast_agg.get_aggregate_nowcast(indicator)
        if nowcast:
            print(f"   {indicator}: {nowcast['aggregate_value']}%")
            print(f"      Spread: ±{nowcast['spread']}%")
            print(f"      Sources: {list(nowcast['sources'].keys())}")
        else:
            print(f"   {indicator}: No data available")

    # 5. Trade candidates (if in a trading window)
    print_header("5. TRADE CANDIDATES")

    if windows:
        updater = BayesianUpdater()

        for w in windows:
            event = w['event']

            if event.event_type == EventType.CPI:
                # Check for CPI divergence
                nowcast = nowcast_agg.get_aggregate_nowcast('cpi_headline')
                if nowcast:
                    # Simulate market price (in real implementation, fetch from Kalshi)
                    market_price = 0.50  # Placeholder

                    evidence = [
                        Evidence(
                            name="cleveland_nowcast",
                            value=nowcast['aggregate_value'],
                            expected=2.7,  # Placeholder threshold
                            timestamp=datetime.now(),
                            reliability=nowcast['confidence']
                        )
                    ]

                    belief = updater.update(
                        market=f"CPI-{event.period}",
                        prior=market_price,
                        evidence=evidence,
                        threshold=2.7  # Placeholder
                    )

                    if abs(belief.edge) > 0.05:
                        print(f"   📈 POTENTIAL TRADE:")
                        print(f"      Market: CPI {event.period}")
                        print(f"      Prior: {belief.prior:.1%}")
                        print(f"      Posterior: {belief.posterior:.1%}")
                        print(f"      Edge: {belief.edge:+.1%}")
                    else:
                        print(f"   No significant edge found for CPI")
    else:
        print("   No active windows - check back closer to release dates")

    # 6. Action items
    print_header("6. TODAY'S ACTION ITEMS")

    actions = []

    # Check if we're approaching a release
    for event in upcoming[:3]:  # Next 3 events
        hours = event.hours_until
        if 24 <= hours <= 72:
            actions.append(f"📋 Prepare for {event.event_type.value.upper()} on {event.release_date}")
        elif hours < 24:
            actions.append(f"🚨 {event.event_type.value.upper()} releases in {hours:.0f} hours - execute or hold")

    if actions:
        for action in actions:
            print(f"   {action}")
    else:
        print("   No immediate actions required")
        print("   Next event:", upcoming[0] if upcoming else "None scheduled")

    print_header("SCAN COMPLETE")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print(f"   Next scan: Run again tomorrow morning\n")


if __name__ == "__main__":
    main()
