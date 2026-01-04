"""
Weather data client combining multiple weather APIs.

Key responsibilities:
1. Fetch forecasts from multiple models (GFS, ECMWF)
2. Fetch official NWS forecast
3. Fetch Visual Crossing commercial forecasts
4. Normalize all forecasts to common format
5. Handle API errors gracefully

Sources:
- Open-Meteo (FREE) - GFS and ECMWF models
- NWS (FREE) - Official US forecasts
- Visual Crossing (API key required) - Commercial accuracy
"""

import time
import requests
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from config.locations import LOCATIONS, get_location
from config.settings import VISUALCROSSING_API_KEY
from utils.logger import get_logger

logger = get_logger("weather_client")


@dataclass
class Forecast:
    """Represents a weather forecast from any source."""
    source: str  # 'ecmwf', 'gfs', 'nws'
    location: str  # Location key (e.g., "NYC")
    forecast_time: datetime  # When the forecast was made
    target_date: date  # Date being forecasted
    high_temp_f: Optional[float]  # High temperature in Fahrenheit
    low_temp_f: Optional[float]  # Low temperature in Fahrenheit
    confidence: Optional[float] = None  # Confidence score if available
    raw_data: Optional[dict] = None  # Raw API response for debugging


@dataclass
class ActualWeather:
    """Represents actual observed weather data."""
    location: str
    date: date
    high_temp_f: float
    low_temp_f: float
    source: str = "nws"


class WeatherClient:
    """
    Client for fetching weather data from multiple sources.

    Combines:
    - Open-Meteo API (GFS and ECMWF models)
    - NWS API (Official US forecasts)
    - Visual Crossing API (Commercial forecasts)
    - Open-Meteo Historical API (for backtesting)
    """

    OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_ECMWF_URL = "https://api.open-meteo.com/v1/ecmwf"  # ECMWF needs separate endpoint
    OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
    NWS_BASE_URL = "https://api.weather.gov"
    VISUALCROSSING_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    def __init__(self):
        """Initialize the weather client."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "WeatherKalshiTrader/1.0 (contact@example.com)",
            "Accept": "application/json"
        })

        # Visual Crossing API key
        self.vc_api_key = VISUALCROSSING_API_KEY

        # Rate limiting
        self._last_request_time = {}
        self._min_intervals = {
            "open_meteo": 0.5,       # 2 requests/second
            "nws": 0.5,              # 2 requests/second
            "visualcrossing": 1.0    # 1 request/second (API limit)
        }

        logger.info("Weather client initialized")

    def _rate_limit(self, api: str):
        """Enforce rate limiting for specific API."""
        last_time = self._last_request_time.get(api, 0)
        min_interval = self._min_intervals.get(api, 0.5)
        elapsed = time.time() - last_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time[api] = time.time()

    def _request(self, url: str, params: dict = None, api: str = "open_meteo",
                 retries: int = 3) -> Optional[dict]:
        """
        Make an API request with error handling and retries.

        Args:
            url: Full URL to request
            params: Query parameters
            api: API identifier for rate limiting
            retries: Number of retries on failure

        Returns:
            Response JSON or None on failure
        """
        self._rate_limit(api)

        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.warning(f"{api} API request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"{api} API request failed after {retries} attempts")
                    return None

        return None

    def get_open_meteo_forecast(self, location_key: str,
                                 models: List[str] = None) -> List[Forecast]:
        """
        Fetch forecasts from Open-Meteo API for multiple models.

        Args:
            location_key: Location identifier (e.g., "NYC")
            models: List of models to fetch. Options: "gfs_seamless", "ecmwf_ifs04"
                   Defaults to both GFS and ECMWF.

        Returns:
            List of Forecast objects, one per model per day
        """
        location = get_location(location_key)
        models = models or ["gfs_seamless", "ecmwf_ifs04"]

        forecasts = []

        for model in models:
            # ECMWF requires a different API endpoint than GFS
            if model == "ecmwf_ifs04":
                url = self.OPEN_METEO_ECMWF_URL
                params = {
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "daily": "temperature_2m_max,temperature_2m_min",
                    "temperature_unit": "fahrenheit",
                    "timezone": location["timezone"]
                }
            else:
                url = self.OPEN_METEO_FORECAST_URL
                params = {
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "daily": "temperature_2m_max,temperature_2m_min",
                    "temperature_unit": "fahrenheit",
                    "timezone": location["timezone"],
                    "forecast_days": 7,
                    "models": model
                }

            response = self._request(url, params=params)

            if not response:
                logger.warning(f"No response from Open-Meteo for model {model}")
                continue

            # Parse the response
            daily_data = response.get("daily", {})
            dates = daily_data.get("time", [])
            highs = daily_data.get("temperature_2m_max", [])
            lows = daily_data.get("temperature_2m_min", [])

            # Map model name to our internal names
            model_name_map = {
                "gfs_seamless": "gfs",
                "ecmwf_ifs04": "ecmwf"
            }
            source = model_name_map.get(model, model)

            forecast_time = datetime.utcnow()

            for i, date_str in enumerate(dates):
                try:
                    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    high = highs[i] if i < len(highs) else None
                    low = lows[i] if i < len(lows) else None

                    if high is not None or low is not None:
                        forecasts.append(Forecast(
                            source=source,
                            location=location_key,
                            forecast_time=forecast_time,
                            target_date=target_date,
                            high_temp_f=high,
                            low_temp_f=low,
                            raw_data={"model": model, "response": response}
                        ))
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing forecast data: {e}")
                    continue

        logger.info(f"Retrieved {len(forecasts)} forecasts from Open-Meteo for {location_key}")
        return forecasts

    def get_nws_forecast(self, location_key: str) -> List[Forecast]:
        """
        Fetch official NWS forecast.

        Args:
            location_key: Location identifier (e.g., "NYC")

        Returns:
            List of Forecast objects
        """
        location = get_location(location_key)
        gridpoint = location["nws_gridpoint"]

        url = f"{self.NWS_BASE_URL}/gridpoints/{gridpoint}/forecast"
        response = self._request(url, api="nws")

        if not response:
            logger.warning(f"No response from NWS for {location_key}")
            return []

        forecasts = []
        forecast_time = datetime.utcnow()

        # NWS returns periods (day/night alternating)
        periods = response.get("properties", {}).get("periods", [])

        # Group periods by date to get high/low
        date_temps = {}

        for period in periods:
            try:
                # Parse the start time to get the date
                start_time = period.get("startTime", "")
                period_date = datetime.fromisoformat(start_time.replace("Z", "+00:00")).date()

                is_daytime = period.get("isDaytime", True)
                temp = period.get("temperature")
                temp_unit = period.get("temperatureUnit", "F")

                # Convert to Fahrenheit if needed
                if temp_unit == "C":
                    temp = temp * 9 / 5 + 32

                if period_date not in date_temps:
                    date_temps[period_date] = {"high": None, "low": None}

                if is_daytime:
                    date_temps[period_date]["high"] = temp
                else:
                    date_temps[period_date]["low"] = temp

            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Error parsing NWS period: {e}")
                continue

        # Create Forecast objects
        for target_date, temps in date_temps.items():
            if temps["high"] is not None or temps["low"] is not None:
                forecasts.append(Forecast(
                    source="nws",
                    location=location_key,
                    forecast_time=forecast_time,
                    target_date=target_date,
                    high_temp_f=temps["high"],
                    low_temp_f=temps["low"],
                    raw_data={"response": response}
                ))

        logger.info(f"Retrieved {len(forecasts)} forecasts from NWS for {location_key}")
        return forecasts

    def get_visualcrossing_forecast(self, location_key: str) -> List[Forecast]:
        """
        Fetch forecast from Visual Crossing API.

        Args:
            location_key: Location identifier (e.g., "NYC")

        Returns:
            List of Forecast objects
        """
        if not self.vc_api_key:
            logger.debug("Visual Crossing API key not configured")
            return []

        location = get_location(location_key)

        # Visual Crossing uses city names
        city_names = {
            "NYC": "New York City",
            "CHI": "Chicago",
            "LA": "Los Angeles",
            "MIA": "Miami",
            "AUS": "Austin",
            "DEN": "Denver",
            "PHI": "Philadelphia"
        }
        city = city_names.get(location_key, location.get("name", location_key))

        # Build URL with location
        url = f"{self.VISUALCROSSING_URL}/{city}"
        params = {
            "unitGroup": "us",
            "key": self.vc_api_key,
            "contentType": "json",
            "include": "days"
        }

        response = self._request(url, params=params, api="visualcrossing")

        if not response:
            logger.warning(f"No response from Visual Crossing for {location_key}")
            return []

        forecasts = []
        forecast_time = datetime.utcnow()

        # Parse the daily forecasts
        days = response.get("days", [])

        for day in days[:7]:  # Get up to 7 days
            try:
                date_str = day.get("datetime", "")
                target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                high = day.get("tempmax")
                low = day.get("tempmin")

                if high is not None or low is not None:
                    forecasts.append(Forecast(
                        source="visualcrossing",
                        location=location_key,
                        forecast_time=forecast_time,
                        target_date=target_date,
                        high_temp_f=high,
                        low_temp_f=low,
                        raw_data={"day": day}
                    ))

            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Error parsing Visual Crossing day: {e}")
                continue

        logger.info(f"Retrieved {len(forecasts)} forecasts from Visual Crossing for {location_key}")
        return forecasts

    def get_all_forecasts(self, location_key: str) -> Dict[str, List[Forecast]]:
        """
        Fetch forecasts from all sources for a location.

        Args:
            location_key: Location identifier (e.g., "NYC")

        Returns:
            Dictionary mapping source names to lists of Forecasts
        """
        all_forecasts = {
            "gfs": [],
            "ecmwf": [],
            "nws": [],
            "visualcrossing": []
        }

        # Get Open-Meteo forecasts (GFS and ECMWF)
        open_meteo_forecasts = self.get_open_meteo_forecast(location_key)
        for forecast in open_meteo_forecasts:
            if forecast.source in all_forecasts:
                all_forecasts[forecast.source].append(forecast)

        # Get NWS forecast
        nws_forecasts = self.get_nws_forecast(location_key)
        all_forecasts["nws"] = nws_forecasts

        # Get Visual Crossing forecast
        vc_forecasts = self.get_visualcrossing_forecast(location_key)
        all_forecasts["visualcrossing"] = vc_forecasts

        return all_forecasts

    def get_forecasts_for_date(self, location_key: str,
                                target_date: date) -> Dict[str, Forecast]:
        """
        Get forecasts from all sources for a specific date.

        Args:
            location_key: Location identifier
            target_date: Date to get forecasts for

        Returns:
            Dictionary mapping source names to Forecast objects
        """
        all_forecasts = self.get_all_forecasts(location_key)

        date_forecasts = {}
        for source, forecasts in all_forecasts.items():
            for forecast in forecasts:
                if forecast.target_date == target_date:
                    date_forecasts[source] = forecast
                    break

        return date_forecasts

    def get_historical_actuals(self, location_key: str, start_date: date,
                                end_date: date) -> List[ActualWeather]:
        """
        Fetch historical actual temperatures from Open-Meteo Archive.

        Args:
            location_key: Location identifier
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            List of ActualWeather objects
        """
        location = get_location(location_key)

        params = {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": "fahrenheit",
            "timezone": location["timezone"]
        }

        response = self._request(self.OPEN_METEO_HISTORICAL_URL, params=params)

        if not response:
            logger.warning(f"No historical data from Open-Meteo for {location_key}")
            return []

        actuals = []
        daily_data = response.get("daily", {})
        dates = daily_data.get("time", [])
        highs = daily_data.get("temperature_2m_max", [])
        lows = daily_data.get("temperature_2m_min", [])

        for i, date_str in enumerate(dates):
            try:
                actual_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                high = highs[i] if i < len(highs) else None
                low = lows[i] if i < len(lows) else None

                if high is not None and low is not None:
                    actuals.append(ActualWeather(
                        location=location_key,
                        date=actual_date,
                        high_temp_f=high,
                        low_temp_f=low,
                        source="open_meteo_archive"
                    ))
            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing historical data: {e}")
                continue

        logger.info(f"Retrieved {len(actuals)} historical records for {location_key}")
        return actuals

    def get_actual_for_date(self, location_key: str,
                            target_date: date) -> Optional[ActualWeather]:
        """
        Get actual weather for a specific date (for settlement).

        Args:
            location_key: Location identifier
            target_date: Date to get actual weather for

        Returns:
            ActualWeather object or None if not available
        """
        # For recent dates, fetch from archive
        actuals = self.get_historical_actuals(location_key, target_date, target_date)

        if actuals:
            return actuals[0]

        return None

    def test_connection(self) -> Dict[str, bool]:
        """
        Test connectivity to all weather APIs.

        Returns:
            Dictionary mapping API names to connection status
        """
        results = {}

        # Test Open-Meteo
        try:
            response = self._request(
                self.OPEN_METEO_FORECAST_URL,
                params={"latitude": 40.7128, "longitude": -74.0060, "daily": "temperature_2m_max"},
                api="open_meteo"
            )
            results["open_meteo"] = response is not None
            logger.info(f"Open-Meteo connection: {'success' if results['open_meteo'] else 'failed'}")
        except Exception as e:
            results["open_meteo"] = False
            logger.error(f"Open-Meteo connection failed: {e}")

        # Test NWS
        try:
            response = self._request(
                f"{self.NWS_BASE_URL}/gridpoints/OKX/33,37/forecast",
                api="nws"
            )
            results["nws"] = response is not None
            logger.info(f"NWS connection: {'success' if results['nws'] else 'failed'}")
        except Exception as e:
            results["nws"] = False
            logger.error(f"NWS connection failed: {e}")

        # Test Visual Crossing
        if self.vc_api_key:
            try:
                response = self._request(
                    f"{self.VISUALCROSSING_URL}/New York City",
                    params={"unitGroup": "us", "key": self.vc_api_key, "contentType": "json"},
                    api="visualcrossing"
                )
                results["visualcrossing"] = response is not None
                logger.info(f"Visual Crossing connection: {'success' if results['visualcrossing'] else 'failed'}")
            except Exception as e:
                results["visualcrossing"] = False
                logger.error(f"Visual Crossing connection failed: {e}")
        else:
            results["visualcrossing"] = False
            logger.info("Visual Crossing: API key not configured")

        return results


# Example usage and testing
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    client = WeatherClient()

    # Test connections
    print("Testing API connections...")
    status = client.test_connection()
    print(f"Open-Meteo: {'✓' if status.get('open_meteo') else '✗'}")
    print(f"NWS: {'✓' if status.get('nws') else '✗'}")

    # Get forecasts for NYC
    print("\nFetching NYC forecasts...")
    forecasts = client.get_all_forecasts("NYC")

    for source, source_forecasts in forecasts.items():
        print(f"\n{source.upper()} Forecasts:")
        for forecast in source_forecasts[:3]:  # Show first 3
            print(f"  {forecast.target_date}: High {forecast.high_temp_f}°F, Low {forecast.low_temp_f}°F")

    # Get historical data
    print("\nFetching historical data...")
    from datetime import timedelta
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)
    actuals = client.get_historical_actuals("NYC", start, end)

    print(f"Historical records: {len(actuals)}")
    for actual in actuals[:3]:
        print(f"  {actual.date}: High {actual.high_temp_f}°F, Low {actual.low_temp_f}°F")
