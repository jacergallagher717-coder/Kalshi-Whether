"""
Kalshi API client for fetching weather market data.

Key responsibilities:
1. Authenticate with Kalshi API
2. Fetch all active weather markets
3. Get order book depth for specific markets
4. Parse market tickers to extract date and strike price

Kalshi Weather Market Ticker Format:
- HIGHNY-25JAN15-T35 = NYC high temp > 35°F on Jan 15, 2025
- LOWNY-25JAN15-T20 = NYC low temp < 20°F on Jan 15, 2025
"""

import time
import base64
import requests
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15, pss
from Crypto.Hash import SHA256

from config.settings import (
    KALSHI_API_KEY, KALSHI_BASE_URL, KALSHI_EMAIL, KALSHI_PASSWORD,
    KALSHI_USE_DEMO, KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH
)
from config.locations import LOCATIONS
from utils.logger import get_logger
from utils.helpers import parse_kalshi_ticker

logger = get_logger("kalshi_client")


@dataclass
class Market:
    """Represents a Kalshi weather market."""
    ticker: str
    title: str
    location: str
    market_type: str  # "high" or "low"
    target_date: date
    temp_threshold: float
    yes_price: float
    no_price: float
    yes_bid: float
    yes_ask: float
    volume: int
    open_interest: int
    status: str
    close_time: Optional[datetime] = None
    is_bracket: bool = False  # True if this is a bracket/range market


@dataclass
class OrderBook:
    """Represents the order book for a market."""
    ticker: str
    timestamp: datetime
    yes_bids: List[Dict[str, float]] = field(default_factory=list)  # [{"price": 0.40, "quantity": 100}, ...]
    yes_asks: List[Dict[str, float]] = field(default_factory=list)
    no_bids: List[Dict[str, float]] = field(default_factory=list)
    no_asks: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class PricePoint:
    """Historical price point for a market."""
    ticker: str
    timestamp: datetime
    yes_price: float
    volume: int


@dataclass
class Order:
    """Represents a Kalshi order."""
    order_id: str
    ticker: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    type: str  # "limit" or "market"
    status: str  # "pending", "open", "filled", "cancelled"
    count: int  # Number of contracts
    price: float  # Limit price
    filled_count: int = 0
    remaining_count: int = 0
    created_at: Optional[datetime] = None


@dataclass
class Position:
    """Represents a position in a market."""
    ticker: str
    market_exposure: int  # Positive = long YES, negative = long NO
    resting_orders_count: int
    total_traded: int
    realized_pnl: float


class KalshiClient:
    """
    Client for interacting with the Kalshi API.

    Handles authentication, rate limiting, and data parsing.
    Supports both API key signing (for demo) and email/password login.
    """

    def __init__(self, api_key: str = None, private_key_path: str = None):
        """
        Initialize the Kalshi client.

        Args:
            api_key: Kalshi API key ID. If not provided, uses KALSHI_API_KEY_ID from settings.
            private_key_path: Path to RSA private key file for signing.
        """
        self.api_key_id = api_key or KALSHI_API_KEY_ID
        self.private_key_path = private_key_path or KALSHI_PRIVATE_KEY_PATH
        self.base_url = KALSHI_BASE_URL
        self.session = requests.Session()
        self.private_key = None
        self.member_id = None

        # Load private key if available (for API key auth)
        self._load_private_key()

        self.session.headers.update({
            "Content-Type": "application/json"
        })

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.05  # 20 requests/second max

        logger.info("Kalshi client initialized")

    def _load_private_key(self):
        """Load RSA private key from file for request signing."""
        try:
            key_path = Path(self.private_key_path)
            if key_path.exists():
                with open(key_path, "rb") as f:
                    self.private_key = RSA.import_key(f.read())
                logger.info("Private key loaded for API signing")
            else:
                logger.debug(f"Private key file not found: {key_path}")
        except Exception as e:
            logger.warning(f"Could not load private key: {e}")

    def _sign_request(self, timestamp: str, method: str, path: str) -> str:
        """
        Sign a request using RSA private key per Kalshi API spec.

        Args:
            timestamp: Unix timestamp in milliseconds as string
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path

        Returns:
            Base64-encoded signature
        """
        if not self.private_key:
            raise ValueError("Private key not loaded - cannot sign request")

        # Kalshi signature format: timestamp + method + path
        message = f"{timestamp}{method}{path}"
        message_hash = SHA256.new(message.encode('utf-8'))

        # Sign with RSA-PSS and SHA256 (Kalshi's required format)
        signature = pss.new(self.private_key).sign(message_hash)

        return base64.b64encode(signature).decode('utf-8')

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _request(self, method: str, endpoint: str, params: dict = None,
                 data: dict = None, retries: int = 3) -> Optional[dict]:
        """
        Make an API request with error handling and retries.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            retries: Number of retries on failure

        Returns:
            Response JSON or None on failure
        """
        self._rate_limit()
        url = f"{self.base_url}{endpoint}"

        # Build headers with API key signature if available
        headers = {}
        if self.private_key and self.api_key_id:
            timestamp = str(int(time.time() * 1000))
            # Full path for signing (includes /trade-api/v2 prefix)
            full_path = f"/trade-api/v2{endpoint}"
            signature = self._sign_request(timestamp, method.upper(), full_path)
            headers = {
                "KALSHI-ACCESS-KEY": self.api_key_id,
                "KALSHI-ACCESS-SIGNATURE": signature,
                "KALSHI-ACCESS-TIMESTAMP": timestamp
            }
            logger.debug(f"Signing: ts={timestamp} method={method.upper()} path={full_path}")

        for attempt in range(retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=30
                )

                if response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None

        return None

    def get_markets(self, status: str = "open", limit: int = 200,
                    cursor: str = None) -> List[dict]:
        """
        Fetch all markets from Kalshi.

        Args:
            status: Market status filter ("open", "closed", "settled")
            limit: Maximum number of markets to return
            cursor: Pagination cursor

        Returns:
            List of market dictionaries
        """
        params = {
            "status": status,
            "limit": limit
        }
        if cursor:
            params["cursor"] = cursor

        response = self._request("GET", "/markets", params=params)

        if response and "markets" in response:
            return response["markets"]
        return []

    def get_weather_markets(self, location: str = None) -> List[Market]:
        """
        Fetch all active weather markets, optionally filtered by location.

        Args:
            location: Location code to filter (e.g., "NYC")

        Returns:
            List of Market objects
        """
        markets = []

        # Map location to series tickers
        series_map = {
            "NYC": ["KXHIGHNY", "KXLOWNY"],
            "CHI": ["KXHIGHCHI", "KXLOWCHI"],
            "LA": ["KXHIGHLAX", "KXLOWLAX"],
            "MIA": ["KXHIGHMIA", "KXLOWMIA"],
            "AUS": ["KXHIGHAUS", "KXLOWAUS"],
            "DEN": ["KXHIGHDEN", "KXLOWDEN"],
            "PHI": ["KXHIGHPHIL", "KXLOWPHIL"],
        }

        # Determine which series to fetch
        if location and location in series_map:
            series_list = series_map[location]
        else:
            # Fetch all weather series
            series_list = []
            for s_list in series_map.values():
                series_list.extend(s_list)

        # Fetch markets from each series
        for series_ticker in series_list:
            try:
                raw_markets = self._get_markets_by_series(series_ticker)

                for market_data in raw_markets:
                    ticker = market_data.get("ticker", "")

                    try:
                        parsed = parse_kalshi_ticker(ticker)
                        market = self._parse_market(market_data, parsed)
                        if market:
                            markets.append(market)
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Skipping market {ticker}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Error fetching series {series_ticker}: {e}")
                continue

        logger.info(f"Found {len(markets)} weather markets" +
                    (f" for {location}" if location else ""))
        return markets

    def _get_markets_by_series(self, series_ticker: str, limit: int = 50) -> List[dict]:
        """Fetch markets for a specific series."""
        params = {
            "series_ticker": series_ticker,
            "limit": limit,
            "status": "open"
        }
        response = self._request("GET", "/markets", params=params)

        if response and "markets" in response:
            return response["markets"]
        return []

    def _parse_market(self, market_data: dict, parsed_ticker: dict) -> Optional[Market]:
        """
        Parse raw market data into a Market object.

        Args:
            market_data: Raw market data from API
            parsed_ticker: Parsed ticker components

        Returns:
            Market object or None if parsing fails
        """
        try:
            # Extract prices - handle different API response formats
            yes_price = market_data.get("yes_price") or market_data.get("last_price") or 0.0
            no_price = market_data.get("no_price") or (1.0 - yes_price) if yes_price else 0.0

            # Convert cents to dollars if needed (Kalshi sometimes uses cents)
            if yes_price > 1:
                yes_price = yes_price / 100
                no_price = no_price / 100

            yes_bid = market_data.get("yes_bid", 0) or 0
            yes_ask = market_data.get("yes_ask", 0) or 0
            if yes_bid > 1:
                yes_bid = yes_bid / 100
            if yes_ask > 1:
                yes_ask = yes_ask / 100

            return Market(
                ticker=parsed_ticker["ticker"],
                title=market_data.get("title", ""),
                location=parsed_ticker["location"],
                market_type=parsed_ticker["market_type"],
                target_date=parsed_ticker["target_date"],
                temp_threshold=parsed_ticker["temp_threshold"],
                yes_price=yes_price,
                no_price=no_price,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                volume=market_data.get("volume", 0) or 0,
                open_interest=market_data.get("open_interest", 0) or 0,
                status=market_data.get("status", "unknown"),
                close_time=self._parse_datetime(market_data.get("close_time")),
                is_bracket=parsed_ticker.get("is_bracket", False)
            )
        except Exception as e:
            logger.error(f"Error parsing market data: {e}")
            return None

    def _parse_datetime(self, dt_string: str) -> Optional[datetime]:
        """Parse datetime string from API response."""
        if not dt_string:
            return None
        try:
            # Handle ISO format
            return datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def get_market(self, ticker: str) -> Optional[Market]:
        """
        Get details for a specific market.

        Args:
            ticker: Market ticker

        Returns:
            Market object or None if not found
        """
        response = self._request("GET", f"/markets/{ticker}")

        if response and "market" in response:
            try:
                parsed = parse_kalshi_ticker(ticker)
                return self._parse_market(response["market"], parsed)
            except ValueError as e:
                logger.error(f"Error parsing ticker {ticker}: {e}")

        return None

    def get_orderbook(self, ticker: str, depth: int = 10) -> Optional[OrderBook]:
        """
        Get order book for a specific market.

        Args:
            ticker: Market ticker
            depth: Number of price levels to retrieve

        Returns:
            OrderBook object or None if not found
        """
        response = self._request("GET", f"/markets/{ticker}/orderbook",
                                 params={"depth": depth})

        if not response:
            return None

        try:
            orderbook = response.get("orderbook", {})
            return OrderBook(
                ticker=ticker,
                timestamp=datetime.utcnow(),
                yes_bids=self._parse_orderbook_side(orderbook.get("yes", {}).get("bids", [])),
                yes_asks=self._parse_orderbook_side(orderbook.get("yes", {}).get("asks", [])),
                no_bids=self._parse_orderbook_side(orderbook.get("no", {}).get("bids", [])),
                no_asks=self._parse_orderbook_side(orderbook.get("no", {}).get("asks", []))
            )
        except Exception as e:
            logger.error(f"Error parsing orderbook for {ticker}: {e}")
            return None

    def _parse_orderbook_side(self, orders: list) -> List[Dict[str, float]]:
        """Parse one side of the order book."""
        parsed = []
        for order in orders:
            price = order.get("price", 0)
            if price > 1:
                price = price / 100
            parsed.append({
                "price": price,
                "quantity": order.get("quantity", 0)
            })
        return parsed

    def get_market_history(self, ticker: str, start_time: datetime = None,
                          end_time: datetime = None) -> List[PricePoint]:
        """
        Get price history for a market.

        Args:
            ticker: Market ticker
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of PricePoint objects
        """
        params = {}
        if start_time:
            params["start_ts"] = int(start_time.timestamp())
        if end_time:
            params["end_ts"] = int(end_time.timestamp())

        response = self._request("GET", f"/markets/{ticker}/history", params=params)

        if not response or "history" not in response:
            return []

        history = []
        for point in response["history"]:
            try:
                price = point.get("yes_price", 0)
                if price > 1:
                    price = price / 100

                history.append(PricePoint(
                    ticker=ticker,
                    timestamp=datetime.fromtimestamp(point.get("ts", 0)),
                    yes_price=price,
                    volume=point.get("volume", 0)
                ))
            except Exception as e:
                logger.debug(f"Error parsing history point: {e}")
                continue

        return history

    def test_connection(self) -> bool:
        """
        Test API connectivity.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self._request("GET", "/exchange/status")
            if response:
                logger.info("Kalshi API connection successful")
                return True
        except Exception as e:
            logger.error(f"Kalshi API connection failed: {e}")

        return False

    # ===================
    # Trading Methods
    # ===================

    def login(self, email: str = None, password: str = None) -> bool:
        """
        Login to Kalshi. Uses API key signing if available, otherwise email/password.

        For API key auth (demo.kalshi.co), no explicit login is needed -
        requests are signed with the private key.

        For email/password auth, gets a session token.

        Args:
            email: Kalshi account email (optional if using API key)
            password: Kalshi account password (optional if using API key)

        Returns:
            True if authentication is ready
        """
        # If we have API key signing, verify it works
        if self.private_key and self.api_key_id:
            logger.info("Using API key authentication (RSA signing)")
            # Test the API key by fetching balance
            try:
                balance = self.get_balance()
                if balance is not None:
                    logger.info(f"API key auth successful! Balance: ${balance.get('balance', 0):.2f}")
                    return True
                else:
                    logger.error("API key auth failed - could not fetch balance")
                    return False
            except Exception as e:
                logger.error(f"API key auth test failed: {e}")
                return False

        # Fall back to email/password login
        email = email or KALSHI_EMAIL
        password = password or KALSHI_PASSWORD

        if not email or not password:
            logger.error("Email and password required for login (no API key configured)")
            return False

        try:
            response = self._request(
                "POST",
                "/login",
                data={"email": email, "password": password}
            )

            if response and "token" in response:
                self.session.headers.update({
                    "Authorization": f"Bearer {response['token']}"
                })
                self.member_id = response.get("member_id")
                logger.info(f"Logged in successfully (demo={KALSHI_USE_DEMO})")
                return True
            else:
                logger.error("Login failed: no token in response")
                return False

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def get_balance(self) -> Optional[Dict]:
        """
        Get account balance.

        Returns:
            Balance info dict with 'balance', 'available_balance', etc.
        """
        response = self._request("GET", "/portfolio/balance")
        if response:
            balance = response.get("balance", 0)
            # Convert cents to dollars if needed
            if balance > 1000:
                balance = balance / 100
            return {
                "balance": balance,
                "available_balance": response.get("available_balance", 0) / 100
                if response.get("available_balance", 0) > 100 else response.get("available_balance", 0)
            }
        return None

    def get_positions(self) -> List[Position]:
        """
        Get all current positions.

        Returns:
            List of Position objects
        """
        response = self._request("GET", "/portfolio/positions")
        if not response or "market_positions" not in response:
            return []

        positions = []
        for pos in response["market_positions"]:
            positions.append(Position(
                ticker=pos.get("ticker", ""),
                market_exposure=pos.get("market_exposure", 0),
                resting_orders_count=pos.get("resting_orders_count", 0),
                total_traded=pos.get("total_traded", 0),
                realized_pnl=pos.get("realized_pnl", 0) / 100 if pos.get("realized_pnl", 0) > 100 else pos.get("realized_pnl", 0)
            ))
        return positions

    def place_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price: float = None,
        order_type: str = "limit"
    ) -> Optional[Order]:
        """
        Place an order on Kalshi.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            price: Limit price (0-1), required for limit orders
            order_type: "limit" or "market"

        Returns:
            Order object if successful
        """
        # Convert price to cents for API
        price_cents = int(price * 100) if price else None

        data = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type
        }

        if order_type == "limit" and price_cents:
            data["yes_price"] = price_cents if side == "yes" else None
            data["no_price"] = price_cents if side == "no" else None

        response = self._request("POST", "/portfolio/orders", data=data)

        if response and "order" in response:
            order_data = response["order"]
            return Order(
                order_id=order_data.get("order_id", ""),
                ticker=ticker,
                side=side,
                action=action,
                type=order_type,
                status=order_data.get("status", "pending"),
                count=count,
                price=price or 0,
                filled_count=order_data.get("filled_count", 0),
                remaining_count=order_data.get("remaining_count", count),
                created_at=datetime.utcnow()
            )

        logger.error(f"Failed to place order: {response}")
        return None

    def buy_yes(self, ticker: str, count: int, limit_price: float) -> Optional[Order]:
        """
        Buy YES contracts at a limit price.

        Args:
            ticker: Market ticker
            count: Number of contracts
            limit_price: Maximum price to pay (0-1)

        Returns:
            Order object if successful
        """
        logger.info(f"Placing BUY YES order: {ticker} x{count} @ ${limit_price:.2f}")
        return self.place_order(ticker, "yes", "buy", count, limit_price, "limit")

    def buy_no(self, ticker: str, count: int, limit_price: float) -> Optional[Order]:
        """
        Buy NO contracts at a limit price.

        Args:
            ticker: Market ticker
            count: Number of contracts
            limit_price: Maximum price to pay (0-1)

        Returns:
            Order object if successful
        """
        logger.info(f"Placing BUY NO order: {ticker} x{count} @ ${limit_price:.2f}")
        return self.place_order(ticker, "no", "buy", count, limit_price, "limit")

    def sell_position(self, ticker: str, side: str, count: int, limit_price: float) -> Optional[Order]:
        """
        Sell existing position.

        Args:
            ticker: Market ticker
            side: "yes" or "no" - which position to sell
            count: Number of contracts to sell
            limit_price: Minimum price to accept (0-1)

        Returns:
            Order object if successful
        """
        logger.info(f"Placing SELL {side.upper()} order: {ticker} x{count} @ ${limit_price:.2f}")
        return self.place_order(ticker, side, "sell", count, limit_price, "limit")

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        response = self._request("DELETE", f"/portfolio/orders/{order_id}")
        if response:
            logger.info(f"Cancelled order {order_id}")
            return True
        return False

    def get_fills(self, ticker: str = None, limit: int = 100) -> List[Dict]:
        """
        Get recent order fills.

        Args:
            ticker: Optional ticker to filter by
            limit: Maximum fills to return

        Returns:
            List of fill records
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker

        response = self._request("GET", "/portfolio/fills", params=params)
        if response and "fills" in response:
            return response["fills"]
        return []

    def execute_signal(self, signal) -> Optional[Order]:
        """
        Execute a trade signal on Kalshi.

        Args:
            signal: TradeSignal object from edge calculator

        Returns:
            Order object if successful
        """
        ticker = signal.ticker
        contracts = signal.recommended_contracts

        if signal.direction == "BUY_YES":
            # Buy YES at the current ask (or slightly above for fills)
            limit_price = min(signal.market_price + 0.02, 0.99)
            return self.buy_yes(ticker, contracts, limit_price)
        else:
            # Buy NO - price is (1 - yes_price)
            no_price = 1.0 - signal.market_price
            limit_price = min(no_price + 0.02, 0.99)
            return self.buy_no(ticker, contracts, limit_price)


# Example usage and testing
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    client = KalshiClient()

    # Test connection
    if client.test_connection():
        print("Connection successful!")

        # Get weather markets
        markets = client.get_weather_markets("NYC")
        print(f"\nFound {len(markets)} NYC weather markets:")

        for market in markets[:5]:  # Show first 5
            print(f"  {market.ticker}: YES @ ${market.yes_price:.2f}, "
                  f"threshold {market.temp_threshold}°F on {market.target_date}")
    else:
        print("Connection failed - check API key")
