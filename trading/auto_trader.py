"""
Automated trading system for weather markets.

This module handles:
1. Continuous market scanning at configured intervals
2. Auto-execution of trades on Kalshi when signals are found
3. Position monitoring with take profit and stop loss
4. Daily trade limits and safety guards
"""

import time
import signal
import sys
import re
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

from config.settings import (
    AUTO_TRADE_ENABLED, AUTO_TRADE_MAX_DAILY_TRADES,
    AUTO_TRADE_MAX_OPEN_POSITIONS, KALSHI_USE_DEMO,
    TRADING_CHECK_INTERVAL_MINUTES, MIN_EDGE_THRESHOLD,
    KALSHI_DEMO_URL, HIGH_EDGE_THRESHOLD, HIGH_EDGE_POSITION_PERCENT,
    NORMAL_POSITION_PERCENT, MAX_CONTRACTS_PER_TRADE
)
from data.collectors.kalshi_client import KalshiClient, Order, Position
from data.collectors.data_manager import DataManager
from trading.signal_generator import SignalGenerator
from trading.paper_trader import PaperTrader
from models.edge_calculator import TradeSignal
from utils.logger import get_logger, setup_trade_logger

logger = get_logger("auto_trader")
trade_logger = setup_trade_logger()


def parse_ticker_date(ticker: str) -> Optional[date]:
    """
    Parse the settlement date from a ticker.

    Format: KXHIGHDEN-26JAN05-T58 -> January 5, 2026

    Args:
        ticker: Market ticker string

    Returns:
        date object or None if parsing fails
    """
    try:
        # Match pattern like 26JAN05 (YY + MON + DD)
        match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})-', ticker)
        if not match:
            return None

        year = int(match.group(1)) + 2000  # 26 -> 2026
        month_str = match.group(2)
        day = int(match.group(3))

        months = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
            'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
            'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        month = months.get(month_str)
        if not month:
            return None

        return date(year, month, day)
    except Exception:
        return None


@dataclass
class ExitStrategy:
    """Configuration for position exit strategies."""
    take_profit_pct: float = 0.30  # Exit if position value up 30%
    stop_loss_pct: float = 0.50    # Exit if position value down 50%
    early_exit_edge: float = 0.05  # Exit early if edge drops below 5%
    max_hold_hours: int = 48       # Maximum hours to hold before settlement


class AutoTrader:
    """
    Automated trading system that scans for opportunities and executes trades.

    Features:
    - Continuous market scanning at configured intervals
    - Auto-execution on Kalshi demo or production
    - Position monitoring with exit strategies
    - Daily limits and safety controls
    """

    def __init__(
        self,
        data_manager: DataManager = None,
        signal_generator: SignalGenerator = None,
        paper_trader: PaperTrader = None,
        exit_strategy: ExitStrategy = None,
        live_trading: bool = False
    ):
        """
        Initialize the auto trader.

        Args:
            data_manager: Data manager with production + demo clients
            signal_generator: Signal generator for market scanning
            paper_trader: Paper trader for local tracking
            exit_strategy: Exit strategy configuration
            live_trading: If True, execute on Kalshi; if False, paper trade only
        """
        self.data_manager = data_manager or DataManager()

        # Use trading_client (demo) for executions, kalshi_client (prod) for data
        self.trading_client = self.data_manager.trading_client
        self.market_client = self.data_manager.kalshi_client

        self.signal_generator = signal_generator or SignalGenerator(data_manager=self.data_manager)
        self.paper_trader = paper_trader or PaperTrader()
        self.exit_strategy = exit_strategy or ExitStrategy()
        self.live_trading = live_trading and AUTO_TRADE_ENABLED

        # State tracking
        self.running = False
        self.trades_today = 0
        self.last_scan_time = None
        self.daily_reset_date = date.today()

        # SAFEGUARD: Track tickers we've already traded this session
        self.traded_tickers_today: set = set()

        # SAFEGUARD: Portfolio protection (conviction is the only trade filter, but protect capital)
        self.MAX_PORTFOLIO_PERCENT = 0.50  # Never deploy more than 50% of portfolio
        self.MAX_SINGLE_TRADE_PERCENT = 0.10  # No single trade > 10% of portfolio

        # For graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Auto trader initialized (live={self.live_trading}, demo={KALSHI_USE_DEMO})")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received, stopping...")
        self.running = False

    def _reset_daily_counters(self):
        """Reset daily counters if it's a new day."""
        if date.today() != self.daily_reset_date:
            self.trades_today = 0
            self.traded_tickers_today.clear()
            self.daily_reset_date = date.today()
            logger.info("Daily counters reset")

    def _calculate_position_size(self, signal: TradeSignal, balance: float) -> int:
        """
        Calculate dynamic position size based on edge and bankroll.

        High-edge trades (50%+) get 5% of bankroll.
        Normal trades get 2% of bankroll.

        Args:
            signal: Trade signal with edge info
            balance: Current account balance

        Returns:
            Number of contracts to trade
        """
        # Determine position percentage based on edge
        if signal.edge >= HIGH_EDGE_THRESHOLD:
            position_pct = HIGH_EDGE_POSITION_PERCENT  # 5% for high edge
            logger.info(f"HIGH EDGE ({signal.edge:.1%}) - using {position_pct:.0%} of bankroll")
        else:
            position_pct = NORMAL_POSITION_PERCENT  # 2% for normal

        # Calculate dollar amount
        position_dollars = balance * position_pct

        # Calculate entry price
        if signal.direction == "BUY_YES":
            entry_price = signal.market_price
        else:
            entry_price = 1.0 - signal.market_price

        # Convert to contracts
        if entry_price > 0:
            contracts = int(position_dollars / entry_price)
        else:
            contracts = 1

        # Apply caps
        contracts = max(1, min(contracts, MAX_CONTRACTS_PER_TRADE))

        logger.debug(f"Position sizing: ${balance:.2f} * {position_pct:.0%} = ${position_dollars:.2f} -> {contracts} contracts @ ${entry_price:.2f}")

        return contracts

    def _can_trade(self) -> tuple[bool, str]:
        """
        Check if we can execute a new trade.

        Note: No arbitrary trade limits - conviction filters in edge_calculator are the gate.
        This only checks portfolio protection.

        Returns:
            Tuple of (can_trade, reason)
        """
        self._reset_daily_counters()

        # PORTFOLIO PROTECTION: Check deployment level
        if self.live_trading:
            try:
                balance = self.trading_client.get_balance()
                if balance:
                    total_portfolio = balance.get('total', 0)
                    cash = balance.get('available', 0) or balance.get('balance', 0)
                    if total_portfolio > 0:
                        deployed_pct = 1 - (cash / total_portfolio)
                        if deployed_pct >= self.MAX_PORTFOLIO_PERCENT:
                            return False, f"PORTFOLIO PROTECTION: {deployed_pct:.0%} deployed (max {self.MAX_PORTFOLIO_PERCENT:.0%})"
            except Exception as e:
                logger.warning(f"Could not check portfolio deployment: {e}")

        return True, "OK"

    def scan_and_execute(self, auto_execute: bool = True) -> List[TradeSignal]:
        """
        Scan markets and optionally execute trades.

        Args:
            auto_execute: If True, automatically execute qualifying trades

        Returns:
            List of signals found
        """
        logger.info("Starting market scan...")
        self.last_scan_time = datetime.now()

        # Scan for signals
        signals = self.signal_generator.scan_markets()

        if not signals:
            logger.info("No trade signals found")
            return []

        logger.info(f"Found {len(signals)} potential signals")

        executed = []
        for signal in signals:
            can_trade, reason = self._can_trade()
            if not can_trade:
                logger.warning(f"Cannot trade: {reason}")
                break

            if auto_execute:
                success = self._execute_trade(signal)
                if success:
                    executed.append(signal)
                    self.trades_today += 1

        if executed:
            logger.info(f"Executed {len(executed)} trades")

        return signals

    def _execute_trade(self, signal: TradeSignal) -> bool:
        """
        Execute a single trade signal.

        Args:
            signal: Trade signal to execute

        Returns:
            True if trade executed successfully
        """
        ticker = signal.ticker

        # SAFEGUARD 1: Check in-memory tracking first (bulletproof)
        if ticker in self.traded_tickers_today:
            logger.info(f"SAFEGUARD: Already traded {ticker} today, skipping")
            return False

        # SAFEGUARD 2: Check if we already have a position in this market
        if self.live_trading:
            # Check real Kalshi positions when live trading
            try:
                kalshi_positions = self.trading_client.get_positions()
                for pos in kalshi_positions:
                    # Use market_exposure (not count) - position exists if exposure > 0
                    if pos.ticker == ticker and pos.market_exposure != 0:
                        logger.info(f"Already have REAL position in {ticker} (exposure: {pos.market_exposure}), skipping")
                        return False
            except Exception as e:
                # FAIL SAFE: If we can't check positions, DON'T TRADE
                logger.error(f"SAFEGUARD: Could not verify positions, refusing to trade: {e}")
                return False
        else:
            # Check paper positions when paper trading
            existing = self.paper_trader.get_position(ticker)
            if existing:
                logger.info(f"Already have paper position in {ticker}, skipping")
                return False

        # DYNAMIC POSITION SIZING based on bankroll and edge
        if self.live_trading:
            try:
                balance_info = self.trading_client.get_balance()
                if balance_info:
                    balance = balance_info.get('available_balance', 100)
                    # Calculate dynamic contracts based on edge and bankroll
                    dynamic_contracts = self._calculate_position_size(signal, balance)
                    signal.recommended_contracts = dynamic_contracts
                    logger.info(f"Dynamic sizing: ${balance:.2f} balance -> {dynamic_contracts} contracts")
            except Exception as e:
                logger.warning(f"Could not get balance for dynamic sizing: {e}")

        # Log the trade attempt
        trade_logger.info(
            f"SIGNAL | {ticker} | {signal.direction} | "
            f"Price: ${signal.market_price:.2f} | Edge: {signal.edge:.1%} | "
            f"Contracts: {signal.recommended_contracts}"
        )

        # Execute on Kalshi if live trading enabled
        kalshi_order = None
        if self.live_trading:
            try:
                kalshi_order = self.trading_client.execute_signal(signal)
                if kalshi_order:
                    trade_logger.info(
                        f"KALSHI ORDER | {kalshi_order.order_id} | {ticker} | "
                        f"Status: {kalshi_order.status}"
                    )
                    # SAFEGUARD: Mark this ticker as traded
                    self.traded_tickers_today.add(ticker)
                    self.trades_today += 1
                else:
                    logger.error(f"Failed to place Kalshi order for {ticker}")
                    return False
            except Exception as e:
                logger.error(f"Error placing Kalshi order: {e}")
                return False

        # Record in paper trader for tracking (if not already tracked via live trade)
        if not self.live_trading:
            paper_trade = self.paper_trader.execute_paper_trade(signal)
            if paper_trade:
                # Track for paper trading too
                self.traded_tickers_today.add(ticker)
                self.trades_today += 1
                logger.info(f"Paper trade executed: {paper_trade.id} - {ticker}")
                return True
            return False

        return True  # Live trade was successful

    def cleanup_stale_orders(self, max_age_hours: int = 6) -> int:
        """
        Cancel stale resting orders that are unlikely to fill.

        Args:
            max_age_hours: Cancel orders older than this many hours

        Returns:
            Number of orders cancelled
        """
        if not self.live_trading:
            return 0

        try:
            all_orders = self.trading_client.get_orders(status="resting")
            cancelled = 0
            now = datetime.utcnow()

            for order in all_orders:
                # Skip if no created_at timestamp
                if not order.created_at:
                    continue

                age_hours = (now - order.created_at).total_seconds() / 3600

                # Cancel if too old OR if price is too low (1-2¢ orders won't fill)
                if age_hours > max_age_hours or order.price <= 0.02:
                    reason = f"stale ({age_hours:.1f}h old)" if age_hours > max_age_hours else f"low price (${order.price:.2f})"
                    logger.info(f"Cancelling {reason} order: {order.ticker} {order.action} {order.side} @ ${order.price:.2f}")
                    if self.trading_client.cancel_order(order.order_id):
                        cancelled += 1

            if cancelled:
                logger.info(f"Cleaned up {cancelled} stale resting orders")
            return cancelled

        except Exception as e:
            logger.warning(f"Error cleaning up stale orders: {e}")
            return 0

    def check_exits(self) -> List[str]:
        """
        Check all open positions for exit conditions.

        Returns:
            List of tickers that were exited
        """
        exited = []
        expired = []
        positions = self.paper_trader.get_open_positions()
        today = date.today()

        for position in positions:
            # Check if market has expired (settlement date is in the past)
            settlement_date = parse_ticker_date(position.ticker)
            if settlement_date and settlement_date < today:
                # Market has already settled - just close paper position, don't try to trade
                logger.info(f"Skipping expired market {position.ticker} (settled {settlement_date})")
                expired.append(position.ticker)
                # Mark paper position as closed (settled by Kalshi)
                self.paper_trader.close_position(position.id, "expired")
                continue

            should_exit, reason = self._should_exit(position)

            if should_exit:
                logger.info(f"Exit triggered for {position.ticker}: {reason}")

                if self.live_trading:
                    self._execute_exit(position)

                exited.append(position.ticker)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired paper positions")

        return exited

    def _should_exit(self, position) -> tuple[bool, str]:
        """
        Check if a position should be exited.

        Args:
            position: Paper trade position

        Returns:
            Tuple of (should_exit, reason)
        """
        # Get current market price from production
        market = self.market_client.get_market(position.ticker)
        if not market:
            return False, ""

        current_price = market.yes_price if position.direction == "BUY_YES" else (1 - market.yes_price)
        entry_price = position.entry_price

        # Calculate P&L percentage
        if entry_price > 0:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = 0

        # Take profit
        if pnl_pct >= self.exit_strategy.take_profit_pct:
            return True, f"Take profit ({pnl_pct:.1%} gain)"

        # Stop loss
        if pnl_pct <= -self.exit_strategy.stop_loss_pct:
            return True, f"Stop loss ({pnl_pct:.1%} loss)"

        # Check if edge has deteriorated
        # Re-analyze the market to get current edge
        try:
            new_signal = self.signal_generator.generate_single_signal(position.ticker)
            if new_signal and abs(new_signal.edge) < self.exit_strategy.early_exit_edge:
                return True, f"Edge deteriorated to {new_signal.edge:.1%}"
        except Exception:
            pass

        # Check max hold time
        hours_held = (datetime.utcnow() - position.created_at).total_seconds() / 3600
        if hours_held >= self.exit_strategy.max_hold_hours:
            return True, f"Max hold time reached ({hours_held:.0f}h)"

        return False, ""

    def _execute_exit(self, position):
        """
        Execute an early exit on Kalshi demo.

        Args:
            position: Position to exit

        Returns:
            True if exit order placed successfully
        """
        ticker = position.ticker

        # SAFEGUARD: Check for existing resting exit orders
        try:
            existing_orders = self.trading_client.get_orders(ticker=ticker, status="resting")
            exit_orders = [o for o in existing_orders if o.action == "sell"]
            if exit_orders:
                logger.info(f"Already have {len(exit_orders)} resting exit orders for {ticker}, skipping")
                return False
        except Exception as e:
            logger.warning(f"Could not check existing orders: {e}")

        # Get market price from production
        market = self.market_client.get_market(ticker)
        if not market:
            logger.error(f"Cannot get market for exit: {ticker}")
            return False

        # Determine exit parameters
        if position.direction == "BUY_YES":
            # Sell YES position
            side = "yes"
            # Sell at bid (or slightly below for fills)
            exit_price = market.yes_bid - 0.01 if market.yes_bid > 0.02 else market.yes_bid
        else:
            # Sell NO position
            side = "no"
            no_bid = 1 - market.yes_ask
            exit_price = no_bid - 0.01 if no_bid > 0.02 else no_bid

        # SAFEGUARD: Don't place orders at absurdly low prices (below 3¢)
        # These won't fill and just create resting order clutter
        MIN_EXIT_PRICE = 0.03
        if exit_price < MIN_EXIT_PRICE:
            logger.warning(f"Exit price ${exit_price:.2f} too low for {ticker} - skipping (would create resting order)")
            logger.info(f"Consider holding {ticker} to settlement or manually exiting")
            return False

        try:
            # Execute on demo trading client
            order = self.trading_client.sell_position(
                ticker,
                side,
                position.contracts,
                exit_price
            )

            if order:
                trade_logger.info(
                    f"EXIT ORDER | {order.order_id} | {ticker} | "
                    f"Sold {position.contracts} @ ${exit_price:.2f}"
                )
                return True
        except Exception as e:
            logger.error(f"Error executing exit: {e}")
        return False

    def run_continuous(self, interval_minutes: int = None):
        """
        Run continuous scanning and trading loop.

        Args:
            interval_minutes: Minutes between scans (default from settings)
        """
        interval = interval_minutes or TRADING_CHECK_INTERVAL_MINUTES

        print(f"\n{'='*60}")
        print("WEATHER TRADING BOT - CONTINUOUS MODE")
        print(f"{'='*60}")
        print(f"Live Trading: {'YES' if self.live_trading else 'NO (Paper Only)'}")
        print(f"Demo Mode: {'YES' if KALSHI_USE_DEMO else 'NO (PRODUCTION!)'}")
        print(f"Scan Interval: {interval} minutes")
        print(f"Max Daily Trades: {AUTO_TRADE_MAX_DAILY_TRADES}")
        print(f"Max Open Positions: {AUTO_TRADE_MAX_OPEN_POSITIONS}")
        print(f"{'='*60}")
        print("Press Ctrl+C to stop\n")

        self.running = True

        # Login to Kalshi demo if live trading
        if self.live_trading:
            if not self.trading_client.login():
                logger.error("Failed to login to Kalshi demo, falling back to paper trading")
                self.live_trading = False

        scan_count = 0
        while self.running:
            try:
                scan_count += 1
                print(f"\n[Scan #{scan_count}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Cleanup stale resting orders every scan
                stale_cancelled = self.cleanup_stale_orders()
                if stale_cancelled:
                    print(f"  Cancelled {stale_cancelled} stale orders")

                # Check for exits first
                exits = self.check_exits()
                if exits:
                    print(f"  Exited positions: {', '.join(exits)}")

                # Scan and execute
                signals = self.scan_and_execute(auto_execute=True)

                if signals:
                    print(f"  Found {len(signals)} signals")
                    for sig in signals[:3]:  # Show top 3
                        print(f"    - {sig.ticker}: {sig.direction} @ ${sig.market_price:.2f}, Edge: {sig.edge:.1%}")
                else:
                    print("  No signals found")

                # Show status
                positions = self.paper_trader.get_open_positions()
                print(f"  Open positions: {len(positions)}")
                print(f"  Trades today: {self.trades_today}/{AUTO_TRADE_MAX_DAILY_TRADES}")

                # Wait for next scan
                print(f"\n  Next scan in {interval} minutes...")

                for _ in range(interval * 60):
                    if not self.running:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

        print("\nTrading bot stopped.")

    def get_status(self) -> Dict:
        """Get current auto trader status."""
        positions = self.paper_trader.get_open_positions()
        summary = self.paper_trader.get_performance_summary()

        return {
            "running": self.running,
            "live_trading": self.live_trading,
            "demo_mode": KALSHI_USE_DEMO,
            "trades_today": self.trades_today,
            "max_daily_trades": AUTO_TRADE_MAX_DAILY_TRADES,
            "open_positions": len(positions),
            "max_open_positions": AUTO_TRADE_MAX_OPEN_POSITIONS,
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "total_trades": summary["total_trades"],
            "win_rate": summary["win_rate"],
            "net_pnl": summary["net_pnl"]
        }


# Example usage
if __name__ == "__main__":
    import argparse
    from utils.logger import setup_logger
    setup_logger()

    parser = argparse.ArgumentParser(description="Weather Market Auto Trader")
    parser.add_argument("--run", action="store_true", help="Run continuous trading loop")
    parser.add_argument("--live", action="store_true", help="Enable live trading on Kalshi demo")
    parser.add_argument("--execute", action="store_true", help="Execute trades on single scan")
    parser.add_argument("--interval", type=int, default=15, help="Scan interval in minutes")
    args = parser.parse_args()

    # Enable live trading if --live flag or AUTO_TRADE_ENABLED is set
    live_mode = args.live or AUTO_TRADE_ENABLED

    print(f"\n{'='*50}")
    print("WEATHER MARKET AUTO TRADER")
    print(f"{'='*50}")
    print(f"Live Trading: {'ENABLED' if live_mode else 'DISABLED (paper only)'}")
    print(f"Demo Mode: {'YES' if KALSHI_USE_DEMO else 'NO - PRODUCTION!'}")
    print(f"{'='*50}\n")

    trader = AutoTrader(live_trading=live_mode)

    if args.run:
        # Run continuous trading loop
        trader.run_continuous(interval_minutes=args.interval)
    else:
        # Single scan
        signals = trader.scan_and_execute(auto_execute=args.execute or live_mode)
        print(f"\nFound {len(signals)} signals")

        if signals:
            print("\nTop signals:")
            for sig in signals[:5]:
                print(f"  {sig.ticker}: {sig.direction} @ ${sig.market_price:.2f} | Edge: {sig.edge:.1%}")

        # Show status
        status = trader.get_status()
        print(f"\nStatus:")
        print(f"  Trades today: {status['trades_today']}/{status['max_daily_trades']}")
        print(f"  Open positions: {status['open_positions']}/{status['max_open_positions']}")
