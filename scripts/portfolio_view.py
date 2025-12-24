#!/usr/bin/env python3
"""
Portfolio view script - shows paper positions with current prices and P&L.
"""

from trading.paper_trader import PaperTrader
from data.collectors.kalshi_client import KalshiClient
from config.settings import KALSHI_PUBLIC_URL
from utils.logger import setup_logger

setup_logger()

def main():
    # Paper trader for positions
    pt = PaperTrader()
    positions = pt.get_open_positions()

    if not positions:
        print("No open positions.")
        return

    # Public API client for current prices
    client = KalshiClient(base_url=KALSHI_PUBLIC_URL, api_key="", private_key_path="")

    print("\n" + "=" * 70)
    print("PAPER PORTFOLIO - LIVE PRICES")
    print("=" * 70)

    total_cost = 0
    total_value = 0
    total_unrealized = 0

    for pos in positions:
        # Get current market price
        market = client.get_market(pos.ticker)

        if market:
            current_price = market.yes_price if pos.direction == "BUY_YES" else (1 - market.yes_price)
        else:
            current_price = pos.entry_price  # Fallback to entry price

        # Calculate P&L
        cost = pos.entry_price * pos.contracts
        value = current_price * pos.contracts
        unrealized_pnl = value - cost
        pnl_pct = (unrealized_pnl / cost * 100) if cost > 0 else 0

        total_cost += cost
        total_value += value
        total_unrealized += unrealized_pnl

        # Color indicator
        indicator = "🟢" if unrealized_pnl >= 0 else "🔴"

        print(f"\n{pos.ticker}")
        print(f"  Direction: {pos.direction}")
        print(f"  Contracts: {pos.contracts}")
        print(f"  Entry:     ${pos.entry_price:.2f}")
        print(f"  Current:   ${current_price:.2f}")
        print(f"  P&L:       ${unrealized_pnl:+.2f} ({pnl_pct:+.1f}%) {indicator}")

    print("\n" + "-" * 70)
    print(f"TOTAL COST:        ${total_cost:.2f}")
    print(f"TOTAL VALUE:       ${total_value:.2f}")
    print(f"UNREALIZED P&L:    ${total_unrealized:+.2f} ({total_unrealized/total_cost*100 if total_cost > 0 else 0:+.1f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
