#!/usr/bin/env python3
"""
Weather Kalshi Paper Trading System - Main Entry Point

Commands:
    python main.py run          # Start the full system with scheduler
    python main.py scan         # One-time market scan
    python main.py status       # Show current positions and P&L
    python main.py settle       # Check and settle positions
    python main.py report       # Generate performance report
    python main.py validate     # Run edge validation analysis
    python main.py backtest     # Run historical backtest (future)

Usage:
    1. Set up your Kalshi API key in .env file
    2. Run: python main.py run
    3. Monitor logs in logs/system.log
"""

import argparse
import sys
import time
from datetime import datetime, date

import schedule

from config.settings import (
    COLLECTION_INTERVAL_MINUTES,
    TRADING_CHECK_INTERVAL_MINUTES,
    KALSHI_API_KEY, KALSHI_USE_DEMO,
    AUTO_TRADE_ENABLED
)
from config.locations import ACTIVE_LOCATIONS
from data.collectors import DataManager
from trading.signal_generator import SignalGenerator
from trading.paper_trader import PaperTrader
from trading.position_manager import PositionManager
from trading.auto_trader import AutoTrader
from analysis.performance import PerformanceAnalyzer
from analysis.edge_validation import EdgeValidator
from analysis.reports import ReportGenerator
from utils.logger import setup_logger, get_logger
from utils.helpers import format_currency, format_percent

# Initialize logger
setup_logger()
logger = get_logger("main")


class TradingSystem:
    """Main trading system coordinator."""

    def __init__(self):
        """Initialize all components."""
        logger.info("Initializing Weather Kalshi Paper Trading System...")

        self.data_manager = DataManager()
        self.signal_generator = SignalGenerator(self.data_manager)
        self.paper_trader = PaperTrader()
        self.position_manager = PositionManager(self.paper_trader, self.data_manager)
        self.analyzer = PerformanceAnalyzer()
        self.validator = EdgeValidator()
        self.report_generator = ReportGenerator()

        logger.info("System initialized successfully")

    def collect_data(self):
        """Collect data from all sources."""
        logger.info("Starting data collection...")
        try:
            summary = self.data_manager.collect_all_data(ACTIVE_LOCATIONS)
            total_forecasts = sum(
                loc.get("forecasts", 0)
                for loc in summary.get("locations", {}).values()
            )
            total_markets = sum(
                loc.get("markets", 0)
                for loc in summary.get("locations", {}).values()
            )
            logger.info(f"Data collection complete: {total_forecasts} forecasts, {total_markets} markets")
        except Exception as e:
            logger.error(f"Data collection failed: {e}")

    def scan_markets(self):
        """Scan markets and execute paper trades on signals."""
        logger.info("Scanning markets for opportunities...")
        try:
            signals = self.signal_generator.scan_markets(ACTIVE_LOCATIONS)

            if not signals:
                logger.info("No trade signals generated")
                return

            logger.info(f"Found {len(signals)} signals")

            # Execute paper trades for each signal
            executed = 0
            for signal in signals:
                # Check if we can open this position
                can_open = self.position_manager.can_open_position(
                    signal.ticker,
                    signal.recommended_contracts,
                    signal.market_price if signal.direction == "BUY_YES" else (1 - signal.market_price)
                )

                if not can_open["allowed"]:
                    logger.debug(f"Cannot open {signal.ticker}: {can_open['reason']}")
                    continue

                # Execute paper trade
                trade = self.paper_trader.execute_paper_trade(signal)
                if trade:
                    executed += 1
                    logger.info(
                        f"Executed: {trade.ticker} {trade.direction} x{trade.contracts} "
                        f"@ {format_currency(trade.entry_price)} | Edge: {format_percent(trade.edge_at_entry)}"
                    )

            logger.info(f"Executed {executed} paper trades")

        except Exception as e:
            logger.error(f"Market scan failed: {e}")

    def check_settlements(self):
        """Check and settle matured positions."""
        logger.info("Checking for positions to settle...")
        try:
            results = self.position_manager.check_settlements()

            if not results:
                logger.info("No positions settled")
                return

            total_pnl = sum(r.get("net_pnl", 0) for r in results)
            wins = sum(1 for r in results if r.get("correct", False))

            logger.info(
                f"Settled {len(results)} positions: {wins} wins, "
                f"{len(results) - wins} losses, {format_currency(total_pnl)} P&L"
            )

        except Exception as e:
            logger.error(f"Settlement check failed: {e}")

    def generate_daily_report(self):
        """Generate and log daily report."""
        logger.info("Generating daily report...")
        try:
            report_text = self.report_generator.generate_daily_text_report()
            logger.info("\n" + report_text)
        except Exception as e:
            logger.error(f"Daily report generation failed: {e}")

    def generate_weekly_report(self):
        """Generate and log weekly report."""
        logger.info("Generating weekly report...")
        try:
            report_text = self.report_generator.generate_weekly_text_report()
            logger.info("\n" + report_text)
        except Exception as e:
            logger.error(f"Weekly report generation failed: {e}")

    def run_scheduled(self):
        """Run the system with scheduled tasks."""
        logger.info("Starting scheduled trading system...")
        logger.info(f"Active locations: {ACTIVE_LOCATIONS}")
        logger.info(f"Data collection interval: {COLLECTION_INTERVAL_MINUTES} minutes")
        logger.info(f"Market scan interval: {TRADING_CHECK_INTERVAL_MINUTES} minutes")

        # Schedule tasks
        schedule.every(COLLECTION_INTERVAL_MINUTES).minutes.do(self.collect_data)
        schedule.every(TRADING_CHECK_INTERVAL_MINUTES).minutes.do(self.scan_markets)
        schedule.every().day.at("06:00").do(self.check_settlements)
        schedule.every().day.at("07:00").do(self.generate_daily_report)
        schedule.every().sunday.at("08:00").do(self.generate_weekly_report)

        # Initial run
        logger.info("Running initial data collection and market scan...")
        self.collect_data()
        self.scan_markets()

        # Main loop
        logger.info("Entering main loop (Ctrl+C to stop)...")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            sys.exit(0)


def cmd_run(args):
    """Run the full trading system."""
    if not KALSHI_API_KEY:
        logger.warning("KALSHI_API_KEY not set - market data will be limited")

    system = TradingSystem()
    system.run_scheduled()


def cmd_scan(args):
    """One-time market scan."""
    system = TradingSystem()

    print("Collecting latest data...")
    system.collect_data()

    print("\nScanning markets...")

    # If verbose, show all markets with analysis
    if getattr(args, 'verbose', False):
        print("\n" + "="*70)
        print("ALL MARKETS ANALYSIS")
        print("="*70)

        for location in ACTIVE_LOCATIONS:
            markets = system.data_manager.get_markets(location)
            if not markets:
                continue

            print(f"\n{location} Markets ({len(markets)} found):")
            print("-"*50)

            for market in markets:
                try:
                    analysis = system.signal_generator.get_market_analysis(market.ticker)

                    if "error" in analysis:
                        print(f"  {market.ticker}: Error - {analysis['error']}")
                        continue

                    signal = analysis.get("signal", {})
                    edge = signal.get("edge", 0) if signal else 0
                    our_prob = signal.get("our_probability", 0) if signal else 0

                    # Show market details
                    edge_str = f"{edge*100:+.1f}%" if edge else "N/A"
                    prob_str = f"{our_prob*100:.0f}%" if our_prob else "N/A"

                    threshold_symbol = ">" if market.market_type == "high" else "<"

                    print(f"  {market.ticker}")
                    print(f"    {market.market_type.upper()} temp {threshold_symbol} {market.temp_threshold}°F on {market.target_date}")
                    print(f"    Market: YES @ ${market.yes_price:.2f} | Our Prob: {prob_str} | Edge: {edge_str}")

                    # Show model forecasts
                    if analysis.get("model_analysis"):
                        temps = [f"{src}: {data['forecast_temp']:.0f}°F"
                                for src, data in analysis["model_analysis"].items()]
                        print(f"    Forecasts: {' | '.join(temps)}")

                    if edge and abs(edge) >= 0.10:
                        print(f"    *** SIGNAL: {signal.get('direction')} ***")
                    print()

                except Exception as e:
                    print(f"  {market.ticker}: Error - {e}")

        print("="*70)
        print(f"Minimum edge for trade: 10%")
        print("="*70)
        return

    signals = system.signal_generator.scan_markets(ACTIVE_LOCATIONS)

    if not signals:
        print("No trade signals found")
        return

    print(f"\nFound {len(signals)} signals:\n")
    for signal in signals:
        print(f"  {signal.ticker}")
        print(f"    Direction: {signal.direction}")
        print(f"    Market Price: {format_currency(signal.market_price)}")
        print(f"    Our Prob: {format_percent(signal.our_probability)}")
        print(f"    Edge: {format_percent(signal.edge)}")
        print(f"    Confidence: {signal.confidence}")
        print(f"    Recommended: {signal.recommended_contracts} contracts")
        print(f"    Reasoning: {signal.reasoning}")
        print()

    if args.execute:
        print("Executing paper trades...")
        system.scan_markets()


def cmd_status(args):
    """Show current status."""
    system = TradingSystem()

    # Position summary
    print(system.position_manager.get_exposure_report())

    # Performance summary
    print("\n" + system.analyzer.generate_summary_text())


def cmd_settle(args):
    """Check and settle positions."""
    system = TradingSystem()

    print("Checking for positions to settle...")
    results = system.position_manager.check_settlements()

    if not results:
        print("No positions were settled")
        return

    print(f"\nSettled {len(results)} positions:\n")
    for r in results:
        status = "WIN" if r["correct"] else "LOSS"
        print(f"  [{status}] {r['ticker']}")
        print(f"    Outcome: {r['outcome']} @ {r['actual_temp']}°F (threshold: {r['threshold']}°F)")
        print(f"    P&L: {format_currency(r['net_pnl'])}")
        print()

    total_pnl = sum(r["net_pnl"] for r in results)
    wins = sum(1 for r in results if r["correct"])
    print(f"Total: {wins} wins, {len(results) - wins} losses, {format_currency(total_pnl)} P&L")


def cmd_report(args):
    """Generate reports."""
    system = TradingSystem()

    if args.weekly:
        print(system.report_generator.generate_weekly_text_report())
    else:
        print(system.report_generator.generate_daily_text_report())


def cmd_validate(args):
    """Run edge validation."""
    validator = EdgeValidator()
    print(validator.generate_validation_report())


def cmd_test_connections(args):
    """Test API connections."""
    print("Testing API connections...\n")

    data_manager = DataManager()
    results = data_manager.test_connections()

    for api, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"  [{symbol}] {api}")

    print()
    if all(results.values()):
        print("All connections successful!")
    else:
        failed = [api for api, status in results.items() if not status]
        print(f"Failed connections: {', '.join(failed)}")


def cmd_analyze_market(args):
    """Analyze a specific market."""
    system = TradingSystem()

    print(f"Analyzing {args.ticker}...\n")

    # Collect fresh data first
    system.collect_data()

    analysis = system.signal_generator.get_market_analysis(args.ticker)

    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    print(f"Market: {analysis['ticker']}")
    print(f"Location: {analysis['location']}")
    print(f"Target Date: {analysis['target_date']} ({analysis['days_out']} days out)")
    print(f"Type: {analysis['market_type']} temp {'>' if analysis['market_type'] == 'high' else '<'} {analysis['temp_threshold']}°F")
    print()

    print("Market Data:")
    md = analysis['market_data']
    print(f"  YES Price: {format_currency(md['yes_price'])}")
    print(f"  YES Bid/Ask: {format_currency(md['yes_bid'])} / {format_currency(md['yes_ask'])}")
    print(f"  Volume: {md['volume']}")
    print(f"  Open Interest: {md['open_interest']}")
    print()

    print(f"Market Implied Temperature: {analysis['market_implied_temp']:.1f}°F")
    print()

    print("Model Forecasts:")
    for model, data in analysis['model_analysis'].items():
        print(f"  {model.upper()}: {data['forecast_temp']:.1f}°F -> {format_percent(data['probability'])} prob")
    print()

    if analysis['signal']:
        sig = analysis['signal']
        print("Signal:")
        print(f"  Direction: {sig['direction']}")
        print(f"  Our Probability: {format_percent(sig['our_probability'])}")
        print(f"  Edge: {format_percent(sig['edge'])}")
        print(f"  Expected Value: {format_currency(sig['expected_value'])}/contract")
        print(f"  Confidence: {sig['confidence']}")
        print()
        print(f"  Reasoning: {sig['reasoning']}")
    else:
        print("No tradeable signal (edge below threshold)")


def cmd_auto(args):
    """Run automated trading bot."""
    from data.collectors.kalshi_client import KalshiClient

    print("\n" + "="*60)
    print("WEATHER TRADING BOT - AUTO MODE")
    print("="*60)

    # Check settings
    if args.live:
        if not AUTO_TRADE_ENABLED:
            print("\nWARNING: AUTO_TRADE_ENABLED is not set in .env")
            print("Set AUTO_TRADE_ENABLED=true to enable live trading")
            print("\nFalling back to paper-only mode...")
            args.live = False
        elif not KALSHI_USE_DEMO:
            print("\n" + "!"*60)
            print("WARNING: You are about to trade with REAL MONEY!")
            print("KALSHI_USE_DEMO=false means PRODUCTION mode")
            print("!"*60)
            confirm = input("\nType 'CONFIRM' to proceed: ")
            if confirm != "CONFIRM":
                print("Aborted.")
                return

    # Create auto trader
    auto_trader = AutoTrader(live_trading=args.live)

    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Live Trading: {'YES' if args.live else 'NO (Paper Only)'}")
    print(f"  Demo Mode: {'YES' if KALSHI_USE_DEMO else 'NO (PRODUCTION)'}")
    print(f"  Scan Interval: {args.interval} minutes")

    if args.once:
        # Single scan and execute
        print("\nRunning single scan...")
        signals = auto_trader.scan_and_execute(auto_execute=not args.dry_run)

        if signals:
            print(f"\nFound {len(signals)} signals:")
            for sig in signals:
                executed = "EXECUTED" if not args.dry_run else "DRY RUN"
                print(f"  [{executed}] {sig.ticker}: {sig.direction} @ ${sig.market_price:.2f}, Edge: {sig.edge:.1%}")
        else:
            print("\nNo signals found")
    else:
        # Run continuous loop
        auto_trader.run_continuous(interval_minutes=args.interval)


def cmd_login(args):
    """Test Kalshi login and show account info."""
    from data.collectors.kalshi_client import KalshiClient

    print("Testing Kalshi login...")
    print(f"Mode: {'DEMO' if KALSHI_USE_DEMO else 'PRODUCTION'}")
    print()

    client = KalshiClient()

    if client.login():
        print("Login successful!")
        print()

        # Get balance
        balance = client.get_balance()
        if balance:
            print(f"Account Balance: ${balance['balance']:.2f}")
            print(f"Available: ${balance['available_balance']:.2f}")

        # Get positions
        positions = client.get_positions()
        if positions:
            print(f"\nOpen Positions: {len(positions)}")
            for pos in positions[:5]:  # Show first 5
                direction = "LONG YES" if pos.market_exposure > 0 else "LONG NO"
                print(f"  {pos.ticker}: {direction} x{abs(pos.market_exposure)}")
        else:
            print("\nNo open positions")
    else:
        print("Login failed!")
        print("Check your KALSHI_EMAIL and KALSHI_PASSWORD in .env")


def cmd_execute(args):
    """Execute a specific trade signal."""
    from data.collectors.kalshi_client import KalshiClient

    system = TradingSystem()
    client = KalshiClient()

    # First, analyze the market
    print(f"Analyzing {args.ticker}...")
    system.collect_data()

    signal = system.signal_generator.generate_single_signal(args.ticker, force=True)

    if not signal:
        print("Could not generate signal for this market")
        return

    print(f"\nSignal for {signal.ticker}:")
    print(f"  Direction: {signal.direction}")
    print(f"  Price: ${signal.market_price:.2f}")
    print(f"  Edge: {signal.edge:.1%}")
    print(f"  Recommended: {signal.recommended_contracts} contracts")

    # Override contracts if specified
    contracts = args.contracts or signal.recommended_contracts

    if args.live:
        print(f"\n{'!'*40}")
        print(f"LIVE TRADE: {signal.direction} {contracts} contracts @ ${signal.market_price:.2f}")
        print(f"{'!'*40}")

        if not KALSHI_USE_DEMO:
            print("\nWARNING: This is REAL MONEY (production mode)")

        confirm = input("\nType 'YES' to execute: ")
        if confirm != "YES":
            print("Aborted.")
            return

        if not client.login():
            print("Login failed!")
            return

        order = client.execute_signal(signal)
        if order:
            print(f"\nOrder placed: {order.order_id}")
            print(f"Status: {order.status}")
        else:
            print("Order failed!")
    else:
        # Paper trade only
        trade = system.paper_trader.execute_paper_trade(signal)
        if trade:
            print(f"\nPaper trade executed: {trade.id}")
        else:
            print("Paper trade failed (may already have position)")


def cmd_dashboard(args):
    """Show real-time P&L dashboard with Kalshi data."""
    from data.collectors.kalshi_client import KalshiClient
    from analysis.model_tracker import ModelTracker

    print("\n" + "="*60)
    print("WEATHER TRADING DASHBOARD")
    print("="*60)

    client = KalshiClient()

    if not client.login():
        print("Login failed! Check credentials.")
        return

    # Account Balance
    balance = client.get_balance()
    if balance:
        print(f"\n💰 ACCOUNT BALANCE")
        print(f"   Total:     ${balance['balance']:.2f}")
        print(f"   Available: ${balance['available_balance']:.2f}")
        print(f"   At Risk:   ${balance['balance'] - balance['available_balance']:.2f}")

    # Open Positions
    positions = client.get_positions()
    if positions:
        print(f"\n📊 OPEN POSITIONS ({len(positions)})")
        print("-" * 50)

        total_exposure = 0
        for pos in positions:
            direction = "YES" if pos.market_exposure > 0 else "NO"
            exposure = abs(pos.market_exposure)
            total_exposure += exposure

            # Extract city from ticker
            ticker_parts = pos.ticker.split('-')
            city = ticker_parts[0].replace('KXHIGH', '').replace('KXLOW', '')

            print(f"   {pos.ticker}")
            print(f"      {direction} x{exposure} contracts | City: {city}")

        print("-" * 50)
        print(f"   Total Exposure: {total_exposure} contracts")
    else:
        print("\n📊 No open positions")

    # Recent fills/trades from paper trader
    system = TradingSystem()
    recent_trades = system.paper_trader.get_all_trades()[:10]  # Get last 10

    if recent_trades:
        print(f"\n📈 RECENT TRADES (last 10)")
        print("-" * 50)

        wins = 0
        losses = 0
        total_pnl = 0

        for trade in recent_trades:
            if trade.status == 'SETTLED':
                result = "WIN" if trade.net_pnl > 0 else "LOSS"
                pnl_str = f"${trade.net_pnl:+.2f}"
                if trade.net_pnl > 0:
                    wins += 1
                else:
                    losses += 1
                total_pnl += trade.net_pnl
                print(f"   [{result}] {trade.ticker} | {pnl_str}")
            else:
                print(f"   [OPEN] {trade.ticker} | Edge: {trade.edge_at_entry:.1%}")

        if wins + losses > 0:
            win_rate = wins / (wins + losses)
            print("-" * 50)
            print(f"   Win Rate: {win_rate:.0%} ({wins}W / {losses}L)")
            print(f"   Total P&L: ${total_pnl:+.2f}")

    # Model Accuracy Report
    if args.models:
        tracker = ModelTracker()
        print("\n" + tracker.generate_report())

    print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Weather Kalshi Paper Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py run              # Start the automated system
    python main.py scan             # One-time market scan
    python main.py scan --execute   # Scan and execute paper trades
    python main.py auto             # Run continuous auto-trading (paper)
    python main.py auto --live      # Run continuous auto-trading (Kalshi)
    python main.py login            # Test Kalshi login
    python main.py execute TICKER   # Execute trade on specific market
    python main.py status           # View positions and performance
    python main.py settle           # Settle matured positions
    python main.py report           # Daily report
    python main.py report --weekly  # Weekly report
    python main.py validate         # Edge validation analysis
    python main.py test             # Test API connections
    python main.py analyze TICKER   # Analyze specific market
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the full trading system")
    run_parser.set_defaults(func=cmd_run)

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="One-time market scan")
    scan_parser.add_argument("--execute", action="store_true", help="Execute paper trades")
    scan_parser.add_argument("--verbose", "-v", action="store_true", help="Show all markets with analysis")
    scan_parser.set_defaults(func=cmd_scan)

    # Auto command (NEW)
    auto_parser = subparsers.add_parser("auto", help="Run automated trading bot")
    auto_parser.add_argument("--live", action="store_true", help="Execute on Kalshi (not just paper)")
    auto_parser.add_argument("--once", action="store_true", help="Single scan then exit")
    auto_parser.add_argument("--dry-run", action="store_true", help="Scan but don't execute")
    auto_parser.add_argument("--interval", type=int, default=15, help="Minutes between scans")
    auto_parser.set_defaults(func=cmd_auto)

    # Login command (NEW)
    login_parser = subparsers.add_parser("login", help="Test Kalshi login and show account")
    login_parser.set_defaults(func=cmd_login)

    # Execute command (NEW)
    execute_parser = subparsers.add_parser("execute", help="Execute trade on specific market")
    execute_parser.add_argument("ticker", help="Market ticker")
    execute_parser.add_argument("--live", action="store_true", help="Execute on Kalshi")
    execute_parser.add_argument("--contracts", type=int, help="Override contract count")
    execute_parser.set_defaults(func=cmd_execute)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show current status")
    status_parser.set_defaults(func=cmd_status)

    # Settle command
    settle_parser = subparsers.add_parser("settle", help="Check and settle positions")
    settle_parser.set_defaults(func=cmd_settle)

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate performance report")
    report_parser.add_argument("--weekly", action="store_true", help="Generate weekly report")
    report_parser.set_defaults(func=cmd_report)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Run edge validation")
    validate_parser.set_defaults(func=cmd_validate)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test API connections")
    test_parser.set_defaults(func=cmd_test_connections)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze specific market")
    analyze_parser.add_argument("ticker", help="Market ticker (e.g., KXHIGHNY-25DEC18-T50)")
    analyze_parser.set_defaults(func=cmd_analyze_market)

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Show real-time P&L dashboard")
    dashboard_parser.add_argument("--models", action="store_true", help="Include model accuracy report")
    dashboard_parser.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
