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
import requests
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from config.settings import KALSHI_API_KEY, KALSHI_BASE_URL
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


class KalshiClient:
    """
    Client for interacting with the Kalshi API.

    Handles authentication, rate limiting, and data parsing.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the Kalshi client.

        Args:
            api_key: Kalshi API key. If not provided, uses KALSHI_API_KEY from settings.
        """
        self.api_key = api_key or KALSHI_API_KEY
        self.base_url = KALSHI_BASE_URL
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.05  # 20 requests/second max

        logger.info("Kalshi client initialized")

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

        for attempt in range(retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
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
