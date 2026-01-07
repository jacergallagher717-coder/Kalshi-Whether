"""
Paper trading execution and tracking system.

This module simulates trades WITHOUT real money:
1. Receives TradeSignal from edge_calculator
2. Validates signal meets minimum criteria
3. "Executes" paper trade at current market price
4. Tracks position until settlement
5. Records outcome and P&L
"""

import sqlite3
import uuid
from datetime import datetime, date
from typing import List, Dict, Optional
from dataclasses import dataclass

from config.settings import (
    TRADES_DB, MIN_EDGE_THRESHOLD, MAX_POSITION_SIZE,
    calculate_kalshi_fee
)
from models.edge_calculator import TradeSignal
from utils.logger import get_logger, setup_trade_logger
from utils.helpers import calculate_pnl, format_currency

logger = get_logger("paper_trader")
trade_logger = setup_trade_logger()


@dataclass
class PaperTrade:
    """Represents a paper trade record."""
    id: str
    created_at: datetime
    ticker: str
    location: str
    target_date: date
    temp_threshold: float
    direction: str
    entry_price: float
    contracts: int
    simulated_fees: float
    our_probability: float
    edge_at_entry: float
    confidence: str
    reasoning: str

    # Settlement fields
    status: str = "OPEN"
    settlement_outcome: Optional[str] = None
    actual_temp: Optional[float] = None
    exit_price: Optional[float] = None
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    settled_at: Optional[datetime] = None


class PaperTrader:
    """
    Manages paper trading execution and position tracking.

    Key functions:
    - Execute paper trades based on signals
    - Track open positions
    - Settle positions when outcomes are known
    - Calculate P&L statistics
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the paper trader.

        Args:
            db_path: Path to trades database
        """
        self.db_path = db_path or TRADES_DB
        self._init_database()
        logger.info("Paper trader initialized")

    def _init_database(self):
        """Ensure database schema exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id TEXT PRIMARY KEY,
                created_at DATETIME NOT NULL,
                ticker TEXT NOT NULL,
                location TEXT NOT NULL,
                target_date DATE NOT NULL,
                temp_threshold REAL NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                contracts INTEGER NOT NULL,
                simulated_fees REAL NOT NULL,
                our_probability REAL NOT NULL,
                edge_at_entry REAL NOT NULL,
                confidence TEXT NOT NULL,
                reasoning TEXT,
                status TEXT DEFAULT 'OPEN',
                settlement_outcome TEXT,
                actual_temp REAL,
                exit_price REAL,
                gross_pnl REAL,
                net_pnl REAL,
                settled_at DATETIME
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_status
            ON paper_trades(status)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_date
            ON paper_trades(target_date)
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at DATETIME NOT NULL,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                market_price REAL NOT NULL,
                our_probability REAL NOT NULL,
                edge REAL NOT NULL,
                confidence TEXT NOT NULL,
                was_executed INTEGER DEFAULT 0,
                execution_reason TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def execute_paper_trade(
        self,
        signal: TradeSignal,
        override_contracts: int = None
    ) -> Optional[PaperTrade]:
        """
        Execute a paper trade based on a signal.

        Validation checks:
        1. Edge >= MIN_EDGE_THRESHOLD
        2. No existing position in same market
        3. Total exposure doesn't exceed limits

        Args:
            signal: TradeSignal from edge calculator
            override_contracts: Override signal's recommended contracts

        Returns:
            PaperTrade record or None if validation fails
        """
        # Validation
        if abs(signal.edge) < MIN_EDGE_THRESHOLD:
            self._log_signal(signal, False, f"Edge {signal.edge:.2%} below threshold")
            return None

        # Check for existing position
        existing = self.get_position(signal.ticker)
        if existing:
            self._log_signal(signal, False, f"Position already exists in {signal.ticker}")
            return None

        # Check exposure limits
        contracts = override_contracts or signal.recommended_contracts
        entry_price = signal.market_price if signal.direction == "BUY_YES" else (1 - signal.market_price)
        total_cost = entry_price * contracts

        if total_cost > MAX_POSITION_SIZE:
            contracts = int(MAX_POSITION_SIZE / entry_price)
            total_cost = entry_price * contracts
            logger.warning(f"Reduced position size to {contracts} contracts to meet limit")

        if contracts < 1:
            self._log_signal(signal, False, "Position size too small")
            return None

        # Calculate fees
        fees = calculate_kalshi_fee(entry_price, contracts)

        # Create trade record
        trade_id = str(uuid.uuid4())[:8]
        trade = PaperTrade(
            id=trade_id,
            created_at=datetime.utcnow(),
            ticker=signal.ticker,
            location=signal.location,
            target_date=signal.target_date,
            temp_threshold=signal.temp_threshold,
            direction=signal.direction,
            entry_price=entry_price,
            contracts=contracts,
            simulated_fees=fees,
            our_probability=signal.our_probability,
            edge_at_entry=signal.edge,
            confidence=signal.confidence,
            reasoning=signal.reasoning
        )

        # Save to database
        self._save_trade(trade)
        self._log_signal(signal, True, f"Executed as trade {trade_id}")

        # Log the trade
        trade_logger.info(
            f"OPEN | {trade.id} | {trade.ticker} | {trade.direction} | "
            f"{trade.contracts} @ {format_currency(trade.entry_price)} | "
            f"Edge: {trade.edge_at_entry:.1%} | Fees: {format_currency(trade.simulated_fees)}"
        )

        logger.info(f"Executed paper trade: {trade.id} - {trade.ticker} {trade.direction}")
        return trade

    def _save_trade(self, trade: PaperTrade):
        """Save trade to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO paper_trades (
                id, created_at, ticker, location, target_date, temp_threshold,
                direction, entry_price, contracts, simulated_fees, our_probability,
                edge_at_entry, confidence, reasoning, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.id,
            trade.created_at.isoformat(),
            trade.ticker,
            trade.location,
            trade.target_date.isoformat(),
            trade.temp_threshold,
            trade.direction,
            trade.entry_price,
            trade.contracts,
            trade.simulated_fees,
            trade.our_probability,
            trade.edge_at_entry,
            trade.confidence,
            trade.reasoning,
            trade.status
        ))

        conn.commit()
        conn.close()

    def _log_signal(self, signal: TradeSignal, executed: bool, reason: str):
        """Log a trade signal to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trade_signals (
                created_at, ticker, direction, market_price, our_probability,
                edge, confidence, was_executed, execution_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            signal.ticker,
            signal.direction,
            signal.market_price,
            signal.our_probability,
            signal.edge,
            signal.confidence,
            1 if executed else 0,
            reason
        ))

        conn.commit()
        conn.close()

    def get_open_positions(self) -> List[PaperTrade]:
        """Get all open positions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM paper_trades WHERE status = 'OPEN'
            ORDER BY target_date ASC
        ''')

        trades = [self._row_to_trade(row) for row in cursor.fetchall()]
        conn.close()

        return trades

    def get_position(self, ticker: str) -> Optional[PaperTrade]:
        """Get open position for a specific ticker."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM paper_trades
            WHERE ticker = ? AND status = 'OPEN'
        ''', (ticker,))

        row = cursor.fetchone()
        conn.close()

        return self._row_to_trade(row) if row else None

    def get_trade(self, trade_id: str) -> Optional[PaperTrade]:
        """Get trade by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM paper_trades WHERE id = ?', (trade_id,))
        row = cursor.fetchone()
        conn.close()

        return self._row_to_trade(row) if row else None

    def get_all_trades(
        self,
        status: str = None,
        start_date: date = None,
        end_date: date = None
    ) -> List[PaperTrade]:
        """Get trades with optional filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = 'SELECT * FROM paper_trades WHERE 1=1'
        params = []

        if status:
            query += ' AND status = ?'
            params.append(status)
        if start_date:
            query += ' AND target_date >= ?'
            params.append(start_date.isoformat())
        if end_date:
            query += ' AND target_date <= ?'
            params.append(end_date.isoformat())

        query += ' ORDER BY created_at DESC'

        cursor.execute(query, params)
        trades = [self._row_to_trade(row) for row in cursor.fetchall()]
        conn.close()

        return trades

    def settle_position(
        self,
        trade_id: str,
        outcome: str,
        actual_temp: float
    ) -> PaperTrade:
        """
        Settle a position with the actual outcome.

        Args:
            trade_id: Trade ID to settle
            outcome: "YES" or "NO"
            actual_temp: Actual temperature recorded

        Returns:
            Updated PaperTrade
        """
        trade = self.get_trade(trade_id)
        if not trade:
            raise ValueError(f"Trade not found: {trade_id}")

        if trade.status != "OPEN":
            raise ValueError(f"Trade already settled: {trade_id}")

        # Determine exit price based on outcome
        if outcome == "YES":
            exit_price = 1.0
        else:
            exit_price = 0.0

        # Calculate P&L
        gross_pnl, net_pnl = calculate_pnl(
            trade.entry_price,
            exit_price if trade.direction == "BUY_YES" else (1 - exit_price),
            trade.contracts,
            trade.direction,
            trade.simulated_fees
        )

        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE paper_trades SET
                status = 'SETTLED',
                settlement_outcome = ?,
                actual_temp = ?,
                exit_price = ?,
                gross_pnl = ?,
                net_pnl = ?,
                settled_at = ?
            WHERE id = ?
        ''', (
            outcome,
            actual_temp,
            exit_price,
            gross_pnl,
            net_pnl,
            datetime.utcnow().isoformat(),
            trade_id
        ))

        conn.commit()
        conn.close()

        # Log settlement
        result = "WIN" if net_pnl > 0 else "LOSS"
        trade_logger.info(
            f"SETTLED | {trade_id} | {trade.ticker} | {result} | "
            f"Outcome: {outcome} | Actual: {actual_temp}°F | "
            f"P&L: {format_currency(net_pnl)}"
        )

        logger.info(
            f"Settled trade {trade_id}: {outcome} @ {actual_temp}°F, "
            f"P&L: {format_currency(net_pnl)}"
        )

        # Return updated trade
        return self.get_trade(trade_id)

    def cancel_position(self, trade_id: str, reason: str = None):
        """Cancel an open position (no P&L effect)."""
        trade = self.get_trade(trade_id)
        if not trade:
            raise ValueError(f"Trade not found: {trade_id}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE paper_trades SET
                status = 'CANCELLED',
                reasoning = reasoning || ' | CANCELLED: ' || ?
            WHERE id = ?
        ''', (reason or "Manual cancellation", trade_id))

        conn.commit()
        conn.close()

        logger.info(f"Cancelled trade {trade_id}: {reason}")

    def close_position(self, trade_id: str, reason: str = "expired"):
        """
        Close an open position for expired markets.

        Used when markets have already settled on Kalshi but we don't have
        the actual temperature data. Marks as EXPIRED to clean up.

        Args:
            trade_id: Trade ID to close
            reason: Reason for closing (default: "expired")
        """
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"Trade not found for close: {trade_id}")
            return

        if trade.status != "OPEN":
            return  # Already closed

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE paper_trades SET
                status = 'EXPIRED',
                settled_at = ?
            WHERE id = ?
        ''', (datetime.utcnow().isoformat(), trade_id))

        conn.commit()
        conn.close()

        logger.info(f"Closed expired position {trade_id} ({trade.ticker}): {reason}")

    def _row_to_trade(self, row) -> PaperTrade:
        """Convert database row to PaperTrade object."""
        if not row:
            return None

        # Get column names from the table schema
        return PaperTrade(
            id=row[0],
            created_at=datetime.fromisoformat(row[1]),
            ticker=row[2],
            location=row[3],
            target_date=date.fromisoformat(row[4]),
            temp_threshold=row[5],
            direction=row[6],
            entry_price=row[7],
            contracts=row[8],
            simulated_fees=row[9],
            our_probability=row[10],
            edge_at_entry=row[11],
            confidence=row[12],
            reasoning=row[13],
            status=row[14],
            settlement_outcome=row[15],
            actual_temp=row[16],
            exit_price=row[17],
            gross_pnl=row[18],
            net_pnl=row[19],
            settled_at=datetime.fromisoformat(row[20]) if row[20] else None
        )

    def get_performance_summary(self) -> Dict:
        """Get overall performance statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get settled trades
        cursor.execute('''
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                SUM(gross_pnl) as gross_pnl,
                SUM(simulated_fees) as total_fees,
                SUM(net_pnl) as net_pnl,
                AVG(edge_at_entry) as avg_edge
            FROM paper_trades
            WHERE status = 'SETTLED'
        ''')

        row = cursor.fetchone()
        conn.close()

        total = row[0] or 0
        wins = row[1] or 0
        losses = row[2] or 0

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0,
            "gross_pnl": row[3] or 0,
            "total_fees": row[4] or 0,
            "net_pnl": row[5] or 0,
            "avg_edge": row[6] or 0,
            "open_positions": len(self.get_open_positions())
        }


# Example usage
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    print("Testing PaperTrader...")

    trader = PaperTrader()

    # Create a mock signal
    from datetime import timedelta

    mock_signal = TradeSignal(
        signal_id="test123",
        timestamp=datetime.utcnow(),
        ticker="HIGHNY-25JAN15-T85",
        location="NYC",
        target_date=date.today() + timedelta(days=1),
        temp_threshold=85.0,
        market_type="high",
        market_price=0.40,
        market_implied_prob=0.40,
        our_probability=0.70,
        edge=0.30,
        expected_value=0.25,
        confidence="high",
        direction="BUY_YES",
        recommended_contracts=10,
        total_cost=4.00,
        potential_profit=5.60,
        potential_loss=4.00,
        kelly_fraction=0.20,
        reasoning="Test trade"
    )

    # Execute paper trade
    trade = trader.execute_paper_trade(mock_signal)
    if trade:
        print(f"\nExecuted trade: {trade.id}")
        print(f"  Ticker: {trade.ticker}")
        print(f"  Direction: {trade.direction}")
        print(f"  Contracts: {trade.contracts}")
        print(f"  Entry: {format_currency(trade.entry_price)}")
        print(f"  Fees: {format_currency(trade.simulated_fees)}")

    # Get open positions
    positions = trader.get_open_positions()
    print(f"\nOpen positions: {len(positions)}")

    # Simulate settlement
    if trade:
        settled = trader.settle_position(trade.id, "YES", 87.0)
        print(f"\nSettled trade {settled.id}:")
        print(f"  Outcome: {settled.settlement_outcome}")
        print(f"  Actual temp: {settled.actual_temp}°F")
        print(f"  Net P&L: {format_currency(settled.net_pnl)}")

    # Performance summary
    summary = trader.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Win rate: {summary['win_rate']:.1%}")
    print(f"  Net P&L: {format_currency(summary['net_pnl'])}")
