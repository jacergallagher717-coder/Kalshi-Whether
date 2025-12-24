#!/usr/bin/env python3
"""
Daily trading routine - run this once or twice per day.

Usage:
    python -m scripts.daily_runner           # Full daily routine
    python -m scripts.daily_runner --scan    # Just scan for signals
    python -m scripts.daily_runner --status  # Just show portfolio
"""

import argparse
from datetime import datetime

from trading.auto_trader import AutoTrader
from trading.paper_trader import PaperTrader
from data.collectors.kalshi_client import KalshiClient
from config.settings import KALSHI_PUBLIC_URL
from utils.logger import setup_logger

setup_logger()


def show_portfolio():
    """Display current portfolio with live prices."""
    pt = PaperTrader()
    positions = pt.get_open_positions()

    if not positions:
        print("\nNo open positions.\n")
        return

    client = KalshiClient(base_url=KALSHI_PUBLIC_URL, api_key="", private_key_path="")

    print("\n" + "=" * 60)
    print("CURRENT PORTFOLIO")
    print("=" * 60)

    total_cost = 0
    total_value = 0

    for pos in positions:
        market = client.get_market(pos.ticker)
        if market:
            current = market.yes_price if pos.direction == "BUY_YES" else (1 - market.yes_price)
        else:
            current = pos.entry_price

        cost = pos.entry_price * pos.contracts
        value = current * pos.contracts
        pnl = value - cost
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0

        total_cost += cost
        total_value += value

        icon = "🟢" if pnl >= 0 else "🔴"
        print(f"{icon} {pos.ticker}: {pos.contracts}x @ ${pos.entry_price:.2f} → ${current:.2f} ({pnl_pct:+.1f}%)")

    total_pnl = total_value - total_cost
    print("-" * 60)
    print(f"Total P&L: ${total_pnl:+.2f} ({total_pnl/total_cost*100 if total_cost > 0 else 0:+.1f}%)")
    print("=" * 60 + "\n")


def show_performance():
    """Display historical performance stats."""
    pt = PaperTrader()
    summary = pt.get_performance_summary()

    print("\n" + "=" * 60)
    print("PERFORMANCE STATS")
    print("=" * 60)
    print(f"Total Trades:    {summary['total_trades']}")
    print(f"Wins:            {summary['wins']}")
    print(f"Losses:          {summary['losses']}")
    print(f"Win Rate:        {summary['win_rate']:.1%}")
    print(f"Net P&L:         ${summary['net_pnl']:.2f}")
    print(f"Open Positions:  {summary['open_positions']}")
    print("=" * 60 + "\n")


def scan_markets():
    """Scan for new trading signals."""
    print("\n" + "=" * 60)
    print(f"MARKET SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    trader = AutoTrader(live_trading=True)
    signals = trader.scan_and_execute(auto_execute=True)

    if signals:
        print(f"\nExecuted {len(signals)} trades:")
        for sig in signals[:5]:
            print(f"  {sig.ticker}: {sig.direction} @ ${sig.market_price:.2f} | Edge: {sig.edge:.1%}")
    else:
        print("\nNo signals met criteria.")

    print("=" * 60 + "\n")


def daily_routine():
    """Full daily trading routine."""
    print("\n" + "=" * 60)
    print(f"DAILY TRADING ROUTINE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. Show current portfolio
    print("\n[1/3] Checking current positions...")
    show_portfolio()

    # 2. Show performance stats
    print("[2/3] Performance summary...")
    show_performance()

    # 3. Scan for new opportunities
    print("[3/3] Scanning for new signals...")
    scan_markets()

    print("Daily routine complete!\n")


def main():
    parser = argparse.ArgumentParser(description="Daily Weather Trading Routine")
    parser.add_argument("--scan", action="store_true", help="Scan and execute trades")
    parser.add_argument("--status", action="store_true", help="Show portfolio status only")
    parser.add_argument("--performance", action="store_true", help="Show performance stats")
    args = parser.parse_args()

    if args.scan:
        scan_markets()
    elif args.status:
        show_portfolio()
    elif args.performance:
        show_performance()
    else:
        daily_routine()


if __name__ == "__main__":
    main()
