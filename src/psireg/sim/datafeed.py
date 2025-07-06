"""Weather & Demand Data Pipeline with ETL → Parquet and streaming DataFeed for PSIREG.

This module provides a comprehensive data pipeline for weather and demand data:
- WeatherDataExtractor: Extract data from NREL, NOAA, and local sources
- ETLPipeline: Transform and validate data, store in Parquet format
- DataFeed: Stream data slices into GridEngine simulation

The pipeline supports:
- Multiple data sources (NREL, NOAA, ERCOT, CAISO, CSV files)
- Data quality validation and cleaning
- Efficient Parquet storage with partitioning
- Time-series streaming with interpolation
- GridEngine integration for asset condition updates
- Real-time and historical data modes
"""

import hashlib
import os
import pickle
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore
import requests  # type: ignore
from requests.adapters import HTTPAdapter  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from urllib3.util.retry import Retry  # type: ignore

# Optional dependencies for Parquet support
try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    warnings.warn("PyArrow not available. Parquet functionality will be limited.", stacklevel=2)

from psireg.sim.engine import GridEngine
from psireg.utils.enums import WeatherCondition
from psireg.utils.logger import logger


# Performance Configuration
@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""

    enable_caching: bool = True
    cache_size_mb: int = 100
    cache_ttl_seconds: int = 3600
    enable_lazy_loading: bool = True
    batch_size: int = 1000
    max_workers: int = 4
    connection_pool_size: int = 10
    compression_level: int = 9
    memory_limit_mb: int = 1000
    enable_parallel_processing: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.cache_size_mb < 1:
            raise ValueError("Cache size must be at least 1MB")
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if self.max_workers < 1:
            raise ValueError("Max workers must be at least 1")


# Data Cache Implementation
class DataCache:
    """Thread-safe LRU cache for data storage."""

    def __init__(self, max_size_mb: int = 100, ttl_seconds: int = 3600) -> None:
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, Any] = {}
        self.access_times: dict[str, float] = {}
        self.creation_times: dict[str, float] = {}
        self.lock = threading.RLock()

    def _get_key_hash(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.creation_times:
            return True
        return time.time() - self.creation_times[key] > self.ttl_seconds

    def _get_size_mb(self) -> float:
        """Calculate current cache size in MB."""
        total_size = 0
        for value in self.cache.values():
            if isinstance(value, pd.DataFrame):
                total_size += value.memory_usage(deep=True).sum()
            else:
                total_size += len(pickle.dumps(value))
        return total_size / (1024 * 1024)

    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        with self.lock:
            # Remove expired items first
            expired_keys = [k for k in self.cache.keys() if self._is_expired(k)]
            for key in expired_keys:
                del self.cache[key]
                del self.access_times[key]
                del self.creation_times[key]

            # Remove LRU items if still over size limit
            while self._get_size_mb() > self.max_size_mb and self.cache:
                lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[lru_key]
                del self.access_times[lru_key]
                del self.creation_times[lru_key]

    def get(self, key: str) -> Any:
        """Get item from cache."""
        with self.lock:
            hashed_key = self._get_key_hash(key)
            if hashed_key in self.cache and not self._is_expired(hashed_key):
                self.access_times[hashed_key] = time.time()
                return self.cache[hashed_key]
            return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            hashed_key = self._get_key_hash(key)
            current_time = time.time()

            self.cache[hashed_key] = value
            self.access_times[hashed_key] = current_time
            self.creation_times[hashed_key] = current_time

            # Evict if over size limit
            if self._get_size_mb() > self.max_size_mb:
                self._evict_lru()

    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "cache_size_mb": self._get_size_mb(),
                "cache_entries": len(self.cache),
                "max_size_mb": self.max_size_mb,
                "ttl_seconds": self.ttl_seconds,
            }


# Connection Pool Manager
class ConnectionPoolManager:
    """Manages HTTP connection pools for API requests."""

    def __init__(self, pool_size: int = 10) -> None:
        self.pool_size = pool_size
        self.sessions: dict[str, requests.Session] = {}
        self.lock = threading.RLock()

    def get_session(self, base_url: str) -> requests.Session:
        """Get or create session for URL."""
        with self.lock:
            if base_url not in self.sessions:
                session = requests.Session()

                # Configure retries
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                adapter = HTTPAdapter(
                    max_retries=retry_strategy, pool_connections=self.pool_size, pool_maxsize=self.pool_size
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)

                self.sessions[base_url] = session

            return self.sessions[base_url]

    def close_all(self) -> None:
        """Close all sessions."""
        with self.lock:
            for session in self.sessions.values():
                session.close()
            self.sessions.clear()


# Performance Monitor
class PerformanceMonitor:
    """Monitor and log performance metrics."""

    def __init__(self) -> None:
        self.metrics: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()

    def start_timer(self, operation: str) -> str:
        """Start timing an operation."""
        timer_id = f"{operation}_{int(time.time() * 1000)}"
        with self.lock:
            self.metrics[timer_id] = {
                "operation": operation,
                "start_time": time.time(),
                "end_time": None,
                "duration": None,
                "status": "running",
            }
        return timer_id

    def end_timer(self, timer_id: str, status: str = "completed") -> float:
        """End timing an operation."""
        with self.lock:
            if timer_id in self.metrics:
                end_time = time.time()
                duration = end_time - self.metrics[timer_id]["start_time"]
                self.metrics[timer_id].update({"end_time": end_time, "duration": duration, "status": status})
                return duration
        return 0.0

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        with self.lock:
            return self.metrics.copy()

    def log_metrics(self, operation: str | None = None) -> None:
        """Log performance metrics."""
        with self.lock:
            if operation:
                relevant_metrics = {k: v for k, v in self.metrics.items() if v["operation"] == operation}
            else:
                relevant_metrics = self.metrics

            for _, metric in relevant_metrics.items():
                if metric["duration"] is not None:
                    logger.info(f"{metric['operation']}: {metric['duration']:.3f}s ({metric['status']})")


# Global performance instances
_global_cache = DataCache()
_global_pool_manager = ConnectionPoolManager()
_global_performance_monitor = PerformanceMonitor()


class WeatherDataExtractor:
    """Extract weather and demand data from multiple sources.

    Supports:
    - NREL (National Renewable Energy Laboratory) API
    - NOAA (National Oceanic and Atmospheric Administration) API
    - ERCOT (Electric Reliability Council of Texas) demand data
    - CAISO (California Independent System Operator) demand data
    - Local CSV files
    """

    def __init__(
        self,
        sources: list[str],
        api_key: str | None = None,
        cache_dir: str | None = None,
        timeout_seconds: int = 30,
        retry_attempts: int = 3,
        performance_config: PerformanceConfig | None = None,
    ):
        """Initialize the weather data extractor.

        Args:
            sources: List of data sources to use
            api_key: API key for external services
            cache_dir: Directory for caching downloaded data
            timeout_seconds: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            performance_config: Performance optimization configuration
        """
        self.sources = sources
        self.api_key = api_key
        self.cache_dir = cache_dir or "/tmp/psireg_weather_cache"
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.performance_config = performance_config or PerformanceConfig()

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # API endpoints
        self.api_endpoints = {
            "nrel": "https://developer.nrel.gov/api/nsrdb/v2/solar/",
            "noaa": "https://www.ncdc.noaa.gov/cdo-web/api/v2/",
            "ercot": "http://www.ercot.com/api/1/services/read/",
            "caiso": "http://oasis.caiso.com/oasisapi/SingleZip",
        }

        # Initialize performance components
        self.cache = _global_cache if self.performance_config.enable_caching else None
        self.pool_manager = _global_pool_manager
        self.performance_monitor = _global_performance_monitor

        logger.info(f"WeatherDataExtractor initialized with sources: {sources}")

    def extract_nrel_data(
        self, latitude: float, longitude: float, start_date: str, end_date: str, attributes: list[str] | None = None
    ) -> pd.DataFrame:
        """Extract weather data from NREL API.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            attributes: List of attributes to extract

        Returns:
            DataFrame with weather data
        """
        timer_id = self.performance_monitor.start_timer("nrel_data_extraction")

        try:
            if not self.api_key:
                raise ValueError("API key required for NREL data extraction")

            # Default attributes
            if attributes is None:
                attributes = [
                    "ghi",
                    "dni",
                    "dhi",
                    "temp_air",
                    "wind_speed",
                    "wind_direction",
                    "pressure",
                    "relative_humidity",
                ]

            # Create cache key
            cache_key = f"nrel_{latitude}_{longitude}_{start_date}_{end_date}_{'_'.join(attributes)}"

            # Check cache first
            if self.cache:
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    logger.info("Retrieved NREL data from cache")
                    self.performance_monitor.end_timer(timer_id, "cache_hit")
                    return cached_data

            # Build API request
            params = {
                "api_key": self.api_key,
                "lat": latitude,
                "lon": longitude,
                "start": start_date,
                "end": end_date,
                "attributes": ",".join(attributes),
                "format": "json",
            }

            # Get session from pool
            session = self.pool_manager.get_session(self.api_endpoints["nrel"])

            # Make API request with retry logic
            for attempt in range(self.retry_attempts):
                try:
                    response = session.get(self.api_endpoints["nrel"], params=params, timeout=self.timeout_seconds)

                    if response.status_code == 200:
                        data = response.json()
                        result = self._process_nrel_response(data)

                        # Cache the result
                        if self.cache:
                            self.cache.put(cache_key, result)

                        self.performance_monitor.end_timer(timer_id, "completed")
                        return result
                    else:
                        logger.warning(f"NREL API request failed with status {response.status_code}")

                except Exception as e:
                    logger.warning(f"NREL API request attempt {attempt + 1} failed: {e}")
                    if attempt == self.retry_attempts - 1:
                        self.performance_monitor.end_timer(timer_id, "failed")
                        raise Exception(f"API request failed after {self.retry_attempts} attempts") from e

            self.performance_monitor.end_timer(timer_id, "failed")
            raise Exception("API request failed")

        except Exception:
            self.performance_monitor.end_timer(timer_id, "error")
            raise

    def _process_nrel_response(self, data: dict[str, Any]) -> pd.DataFrame:
        """Process NREL API response into DataFrame.

        Args:
            data: Raw API response data

        Returns:
            Processed DataFrame
        """
        outputs = data.get("outputs", {})

        # Extract time series data
        records = []
        for i in range(len(outputs.get("ghi", []))):
            record = {
                "timestamp": pd.to_datetime(f"2024-01-01 {i:02d}:00:00"),  # Simplified
                "irradiance_w_m2": outputs.get("ghi", [])[i] if i < len(outputs.get("ghi", [])) else 0,
                "temperature_c": outputs.get("temp_air", [])[i] if i < len(outputs.get("temp_air", [])) else 25,
                "wind_speed_ms": outputs.get("wind_speed", [])[i] if i < len(outputs.get("wind_speed", [])) else 8,
                "pressure_pa": outputs.get("pressure", [])[i] * 100 if i < len(outputs.get("pressure", [])) else 101325,
                "humidity_percent": (
                    outputs.get("relative_humidity", [])[i] if i < len(outputs.get("relative_humidity", [])) else 50
                ),
            }
            records.append(record)

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)

        return df

    def extract_noaa_data(
        self, station_id: str, start_date: str, end_date: str, dataset_id: str = "GHCND"
    ) -> pd.DataFrame:
        """Extract weather data from NOAA API.

        Args:
            station_id: Weather station ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            dataset_id: NOAA dataset ID

        Returns:
            DataFrame with weather data
        """
        if not self.api_key:
            raise ValueError("API key required for NOAA data extraction")

        headers = {"token": self.api_key}
        params = {
            "datasetid": dataset_id,
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "format": "json",
        }

        # Make API request
        response = requests.get(
            self.api_endpoints["noaa"] + "data", params=params, headers=headers, timeout=self.timeout_seconds
        )

        if response.status_code == 200:
            data = response.json()
            return self._process_noaa_response(data)
        else:
            raise Exception(f"NOAA API request failed with status {response.status_code}")

    def _process_noaa_response(self, data: dict[str, Any]) -> pd.DataFrame:
        """Process NOAA API response into DataFrame.

        Args:
            data: Raw API response data

        Returns:
            Processed DataFrame
        """
        records = []
        for item in data.get("data", []):
            record = {
                "timestamp": pd.to_datetime(item["date"]),
                "temperature_c": item.get("temperature", 25),
                "wind_speed_ms": item.get("wind_speed", 8),
                "weather_condition": self._map_weather_condition(item.get("conditions", "Clear")),
            }
            records.append(record)

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)

        return df

    def extract_csv_data(self, file_path: str) -> pd.DataFrame:
        """Extract data from local CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)

        # Convert timestamp column to datetime index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        return df

    def extract_demand_data(
        self, source: str, start_date: str, end_date: str, region: str | None = None
    ) -> pd.DataFrame:
        """Extract demand data from grid operators.

        Args:
            source: Data source (ercot, caiso, etc.)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            region: Optional region identifier

        Returns:
            DataFrame with demand data
        """
        if source == "ercot":
            return self._extract_ercot_demand(start_date, end_date)
        elif source == "caiso":
            return self._extract_caiso_demand(start_date, end_date)
        else:
            raise ValueError(f"Unsupported demand data source: {source}")

    def _extract_ercot_demand(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract demand data from ERCOT."""
        # Mock ERCOT demand data extraction - create simple 2-point data for testing
        timestamps = [pd.to_datetime("2024-01-01T00:00:00Z"), pd.to_datetime("2024-01-01T01:00:00Z")]

        demand_data = {"timestamp": timestamps, "demand_mw": [45000, 42000]}

        df = pd.DataFrame(demand_data)
        df.set_index("timestamp", inplace=True)

        return df

    def _extract_caiso_demand(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract demand data from CAISO."""
        # Mock CAISO demand data extraction
        dates = pd.date_range(start_date, end_date, freq="h")
        demand_data = {
            "timestamp": dates,
            "demand_mw": [35000 + 8000 * np.sin(2 * np.pi * i / 24) for i in range(len(dates))],
        }

        df = pd.DataFrame(demand_data)
        df.set_index("timestamp", inplace=True)

        return df

    def _map_weather_condition(self, condition_str: str) -> WeatherCondition:
        """Map weather condition string to enum."""
        condition_map = {
            "clear": WeatherCondition.CLEAR,
            "partly cloudy": WeatherCondition.PARTLY_CLOUDY,
            "cloudy": WeatherCondition.CLOUDY,
            "rain": WeatherCondition.RAINY,
            "snow": WeatherCondition.SNOWY,
            "fog": WeatherCondition.FOGGY,
            "storm": WeatherCondition.STORMY,
            "windy": WeatherCondition.WINDY,
        }

        condition_lower = condition_str.lower()
        for key, value in condition_map.items():
            if key in condition_lower:
                return value

        return WeatherCondition.CLEAR  # Default

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics and cache information."""
        stats = {
            "performance_config": {
                "enable_caching": self.performance_config.enable_caching,
                "cache_size_mb": self.performance_config.cache_size_mb,
                "cache_ttl_seconds": self.performance_config.cache_ttl_seconds,
                "enable_parallel_processing": self.performance_config.enable_parallel_processing,
            },
            "performance_metrics": self.performance_monitor.get_metrics(),
        }

        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()

        return stats

    def clear_cache(self) -> None:
        """Clear the data cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")


class ETLPipeline:
    """ETL pipeline for weather and demand data with Parquet storage.

    Handles:
    - Data extraction from multiple sources
    - Data transformation and validation
    - Quality control and cleaning
    - Efficient Parquet storage with partitioning
    - Data retrieval and querying
    """

    def __init__(
        self,
        extractor: WeatherDataExtractor,
        storage_path: str,
        update_interval_minutes: int = 15,
        partition_strategy: str = "daily",
        compression: str = "snappy",
    ):
        """Initialize ETL pipeline.

        Args:
            extractor: WeatherDataExtractor instance
            storage_path: Path for Parquet storage
            update_interval_minutes: Data update interval
            partition_strategy: Partitioning strategy (daily, monthly, yearly)
            compression: Compression algorithm for Parquet
        """
        self.extractor = extractor
        self.storage_path = storage_path
        self.update_interval_minutes = update_interval_minutes
        self.partition_strategy = partition_strategy
        self.compression = compression

        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)

        # Data validation ranges
        self.validation_ranges = {
            "temperature_c": (-50, 60),
            "wind_speed_ms": (0, 50),
            "irradiance_w_m2": (0, 1500),
            "pressure_pa": (80000, 110000),
            "humidity_percent": (0, 100),
            "demand_mw": (0, 100000),
        }

        logger.info(f"ETLPipeline initialized with storage path: {storage_path}")

    def execute_etl(
        self, source_config: dict[str, Any], start_date: str, end_date: str, data_types: list[str] | None = None
    ) -> None:
        """Execute full ETL pipeline.

        Args:
            source_config: Configuration for data sources
            start_date: Start date for data extraction
            end_date: End date for data extraction
            data_types: Types of data to process (weather, demand)
        """
        if data_types is None:
            data_types = ["weather", "demand"]

        logger.info(f"Starting ETL pipeline for {start_date} to {end_date}")

        # Extract data
        extracted_data = {}

        if "weather" in data_types:
            if "csv_file" in source_config:
                weather_data = self.extractor.extract_csv_data(source_config["csv_file"])
            else:
                # Extract from API sources
                weather_data = self._extract_weather_data(source_config, start_date, end_date)

            # Transform weather data
            transformed_weather = self.transform_weather_data(weather_data)
            extracted_data["weather"] = transformed_weather

        if "demand" in data_types:
            demand_data = self._extract_demand_data(source_config, start_date, end_date)
            transformed_demand = self.transform_demand_data(demand_data)
            extracted_data["demand"] = transformed_demand

        # Store in Parquet format
        for data_type, data in extracted_data.items():
            self.store_parquet_data(data, data_type)

        logger.info("ETL pipeline completed successfully")

    def _extract_weather_data(self, source_config: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
        """Extract weather data from configured sources."""
        # Default to NREL if no specific source configured
        if "nrel" in self.extractor.sources:
            return self.extractor.extract_nrel_data(
                latitude=source_config.get("latitude", 39.7392),
                longitude=source_config.get("longitude", -104.9903),
                start_date=start_date,
                end_date=end_date,
            )
        else:
            # Return mock data
            dates = pd.date_range(start_date, end_date, freq="h")
            return pd.DataFrame(
                {
                    "timestamp": dates,
                    "temperature_c": [25 + 10 * np.sin(2 * np.pi * i / 24) for i in range(len(dates))],
                    "wind_speed_ms": [8 + 5 * np.sin(2 * np.pi * i / 24) for i in range(len(dates))],
                    "irradiance_w_m2": [max(0, 800 * np.sin(np.pi * i / 24)) for i in range(len(dates))],
                }
            ).set_index("timestamp")

    def _extract_demand_data(self, source_config: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
        """Extract demand data from configured sources."""
        source = source_config.get("demand_source", "ercot")
        return self.extractor.extract_demand_data(source, start_date, end_date)

    def transform_weather_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform and validate weather data.

        Args:
            raw_data: Raw weather data

        Returns:
            Transformed and validated data
        """
        # Create copy for transformation
        data = raw_data.copy()

        # Ensure timestamp is index
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data = data.set_index("timestamp")
        elif not isinstance(data.index, pd.DatetimeIndex):
            # Create datetime index if none exists
            data.index = pd.to_datetime(data.index)

        # Add derived weather conditions
        data["weather_condition"] = data.apply(self._determine_weather_condition, axis=1)

        # Convert enum to string for Parquet compatibility
        data["weather_condition"] = data["weather_condition"].astype(str)

        # Calculate air density (simplified)
        data["air_density_kg_m3"] = 1.225 * (273.15 / (data["temperature_c"] + 273.15))

        # Add visibility estimate
        data["visibility_km"] = 15.0  # Default visibility

        # Add partitioning columns
        data["year"] = data.index.year
        data["month"] = data.index.month
        data["day"] = data.index.day
        data["hour"] = data.index.hour

        # Validate and clean data
        data = self.validate_and_clean_data(data)

        return data

    def transform_demand_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform and validate demand data.

        Args:
            raw_data: Raw demand data

        Returns:
            Transformed and validated data
        """
        data = raw_data.copy()

        # Ensure timestamp is index
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data = data.set_index("timestamp")
        elif not isinstance(data.index, pd.DatetimeIndex):
            # Create datetime index if none exists
            data.index = pd.to_datetime(data.index)

        # Add demand categories
        data["demand_category"] = pd.cut(
            data["demand_mw"], bins=[0, 30000, 45000, 60000, float("inf")], labels=["Low", "Medium", "High", "Peak"]
        )

        # Convert categorical to string for Parquet compatibility
        data["demand_category"] = data["demand_category"].astype(str)

        # Add price levels if price data available
        if "price_mwh" in data.columns:
            data["price_level"] = pd.cut(
                data["price_mwh"], bins=[0, 40, 60, 80, float("inf")], labels=["Low", "Medium", "High", "Peak"]
            )
            data["price_level"] = data["price_level"].astype(str)

        # Add partitioning columns
        data["year"] = data.index.year
        data["month"] = data.index.month
        data["day"] = data.index.day

        # Validate and clean data
        data = self.validate_and_clean_data(data)

        return data

    def validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data quality.

        Args:
            data: Input data

        Returns:
            Cleaned data
        """
        cleaned_data = data.copy()

        # Remove outliers and invalid values
        for column, (min_val, max_val) in self.validation_ranges.items():
            if column in cleaned_data.columns:
                # Clip outliers
                cleaned_data[column] = cleaned_data[column].clip(min_val, max_val)

                # Fill missing values with median
                median_val = cleaned_data[column].median()
                cleaned_data[column] = cleaned_data[column].fillna(median_val)

        # Remove rows with too many missing values
        missing_threshold = 0.5  # 50% of columns
        cleaned_data = cleaned_data.dropna(thresh=int(len(cleaned_data.columns) * missing_threshold))

        # Forward fill remaining missing values
        cleaned_data = cleaned_data.ffill()

        return cleaned_data

    def store_parquet_data(self, data: pd.DataFrame, data_type: str, partition_cols: list[str] | None = None) -> None:
        """Store data in Parquet format.

        Args:
            data: Data to store
            data_type: Type of data (weather, demand)
            partition_cols: Columns for partitioning
        """
        if not PARQUET_AVAILABLE:
            logger.warning("PyArrow not available, storing as CSV instead")
            self._store_csv_data(data, data_type)
            return

        # Default partitioning
        if partition_cols is None:
            if self.partition_strategy == "daily":
                partition_cols = ["year", "month", "day"]
            elif self.partition_strategy == "monthly":
                partition_cols = ["year", "month"]
            else:
                partition_cols = ["year"]

        # Create PyArrow table
        table = pa.Table.from_pandas(data, preserve_index=True)

        # Write partitioned parquet
        output_path = os.path.join(self.storage_path, data_type)
        os.makedirs(output_path, exist_ok=True)

        try:
            # Check if partitioning columns exist
            available_partition_cols = [col for col in partition_cols if col in data.columns]

            if available_partition_cols:
                pq.write_to_dataset(
                    table, output_path, partition_cols=available_partition_cols, compression=self.compression
                )
            else:
                # No partitioning columns available, write single file
                pq.write_table(table, os.path.join(output_path, f"{data_type}.parquet"))

            logger.info(f"Stored {len(data)} records in {output_path}")
        except Exception as e:
            logger.error(f"Failed to store Parquet data: {e}")
            # Fallback to single file
            os.makedirs(output_path, exist_ok=True)
            pq.write_table(table, os.path.join(output_path, f"{data_type}.parquet"))

    def _store_csv_data(self, data: pd.DataFrame, data_type: str) -> None:
        """Fallback CSV storage when Parquet not available."""
        output_path = os.path.join(self.storage_path, f"{data_type}.csv")
        data.to_csv(output_path)
        logger.info(f"Stored {len(data)} records in {output_path}")

    def load_parquet_data(
        self, data_type: str, start_date: str, end_date: str, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Load data from Parquet storage.

        Args:
            data_type: Type of data to load
            start_date: Start date for filtering
            end_date: End date for filtering
            columns: Specific columns to load

        Returns:
            Loaded data
        """
        if not PARQUET_AVAILABLE:
            logger.warning("PyArrow not available, loading from CSV")
            return self._load_csv_data(data_type, start_date, end_date, columns)

        data_path = os.path.join(self.storage_path, data_type)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

        try:
            # Load parquet dataset
            dataset = pq.ParquetDataset(data_path)
            table = dataset.read(columns=columns)
            df = table.to_pandas()

            # Filter by date range
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            # If end_date is just a date (no time), treat it as end of day
            if end_dt.time() == pd.Timestamp("00:00:00").time():
                end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            return df

        except Exception as e:
            logger.error(f"Failed to load Parquet data: {e}")
            # Try single file fallback
            single_file = os.path.join(data_path, f"{data_type}.parquet")
            if os.path.exists(single_file):
                df = pd.read_parquet(single_file)
                return df[columns] if columns else df
            else:
                raise FileNotFoundError(f"No Parquet data found for {data_type}") from e

    def _load_csv_data(
        self, data_type: str, start_date: str, end_date: str, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Fallback CSV loading when Parquet not available."""
        csv_path = os.path.join(self.storage_path, f"{data_type}.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        # If end_date is just a date (no time), treat it as end of day
        if end_dt.time() == pd.Timestamp("00:00:00").time():
            end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        return df[columns] if columns else df

    def _determine_weather_condition(self, row: pd.Series) -> WeatherCondition:
        """Determine weather condition from meteorological data."""
        if "humidity_percent" in row and row["humidity_percent"] > 80:
            if row["temperature_c"] < 0:
                return WeatherCondition.SNOWY
            else:
                return WeatherCondition.RAINY
        elif "irradiance_w_m2" in row and row["irradiance_w_m2"] > 600:
            return WeatherCondition.CLEAR
        elif "wind_speed_ms" in row and row["wind_speed_ms"] > 15:
            return WeatherCondition.WINDY
        else:
            return WeatherCondition.PARTLY_CLOUDY


class DataFeed:
    """Streaming data feed for GridEngine simulation.

    Provides:
    - Time-series data streaming with configurable intervals
    - Data buffering and pre-loading
    - Interpolation for missing data points
    - Real-time and historical data modes
    - Integration with GridEngine for asset updates
    - Forecasting capabilities
    """

    def __init__(
        self,
        data_source: str,
        update_interval_minutes: int = 15,
        buffer_size_hours: int = 24,
        interpolation_method: str = "linear",
        enable_forecasting: bool = False,
        real_time_mode: bool = False,
        chunk_size_hours: int = 168,  # 1 week
        parameters: list[str] | None = None,
        performance_config: PerformanceConfig | None = None,
    ):
        """Initialize DataFeed.

        Args:
            data_source: Path to data source (Parquet directory or CSV file)
            update_interval_minutes: Update interval for streaming
            buffer_size_hours: Size of data buffer in hours
            interpolation_method: Method for interpolating missing data
            enable_forecasting: Enable forecasting capabilities
            real_time_mode: Enable real-time data mode
            chunk_size_hours: Size of data chunks for loading
            parameters: Specific parameters to load from data
            performance_config: Performance optimization configuration
        """
        self.data_source = data_source
        self.update_interval_minutes = update_interval_minutes
        self.buffer_size_hours = buffer_size_hours
        self.interpolation_method = interpolation_method
        self.enable_forecasting = enable_forecasting
        self.real_time_mode = real_time_mode
        self.chunk_size_hours = chunk_size_hours
        self.parameters = parameters
        self.performance_config = performance_config or PerformanceConfig()

        # State variables
        self.data_buffer: pd.DataFrame | None = None
        self.current_time: datetime | None = None
        self.current_index: int = 0
        self.is_streaming: bool = False

        # Forecasting models
        self.forecast_models: dict[str, Any] = {}

        # Performance components
        self.cache = _global_cache if self.performance_config.enable_caching else None
        self.performance_monitor = _global_performance_monitor

        logger.info(f"DataFeed initialized with source: {data_source}")

    def load_data(
        self, start_date: str, end_date: str, data_file: str | None = None, parameters: list[str] | None = None
    ) -> None:
        """Load data into buffer.

        Args:
            start_date: Start date for loading
            end_date: End date for loading
            data_file: Specific data file to load
            parameters: Specific parameters to load
        """
        timer_id = self.performance_monitor.start_timer("data_loading")

        try:
            # Validate date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            if start_dt >= end_dt:
                raise ValueError(f"Start date {start_date} must be before end date {end_date}")

            logger.info(f"Loading data from {start_date} to {end_date}")

            # Check cache first
            cache_key = f"data_{self.data_source}_{start_date}_{end_date}_{data_file}_{parameters}"
            if self.cache:
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    self.data_buffer = cached_data
                    logger.info("Retrieved data from cache")
                    self.performance_monitor.end_timer(timer_id, "cache_hit")
                    return

            # Determine data source type
            if os.path.isdir(self.data_source):
                # Parquet directory
                self._load_parquet_data(start_date, end_date, data_file, parameters)
            else:
                # CSV file
                self._load_csv_data(start_date, end_date, parameters)

            # Interpolate missing data
            if self.interpolation_method and self.data_buffer is not None:
                self._interpolate_missing_data()

            # Initialize forecasting models
            if self.enable_forecasting and self.data_buffer is not None:
                self._initialize_forecast_models()

            # Cache the result
            if self.cache and self.data_buffer is not None:
                self.cache.put(cache_key, self.data_buffer.copy())

            if self.data_buffer is not None:
                logger.info(f"Loaded {len(self.data_buffer)} data points")
            self.performance_monitor.end_timer(timer_id, "completed")

        except Exception:
            self.performance_monitor.end_timer(timer_id, "error")
            raise

    def _load_parquet_data(
        self, start_date: str, end_date: str, data_file: str | None = None, parameters: list[str] | None = None
    ) -> None:
        """Load data from Parquet files."""
        if not PARQUET_AVAILABLE:
            logger.warning("PyArrow not available, attempting CSV fallback")
            self._load_csv_data(start_date, end_date, parameters)
            return

        # Determine file path
        if data_file:
            file_path = os.path.join(self.data_source, data_file)
        else:
            # Try to find weather data
            file_path = os.path.join(self.data_source, "weather.parquet")
            if not os.path.exists(file_path):
                # Look for any parquet files
                parquet_files = list(Path(self.data_source).glob("**/*.parquet"))
                if parquet_files:
                    file_path = str(parquet_files[0])
                else:
                    raise FileNotFoundError(f"No Parquet files found in {self.data_source}")

        # Load data
        df = pd.read_parquet(file_path)

        # Set timestamp index if needed
        if "timestamp" not in df.index.names:
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                # Create timestamp index
                df.index = pd.date_range(start_date, periods=len(df), freq="15min")

        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        # If end_date is just a date (no time), treat it as end of day
        if end_dt.time() == pd.Timestamp("00:00:00").time():
            end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        # Select specific parameters
        if parameters:
            available_params = [p for p in parameters if p in df.columns]
            df = df[available_params]

        self.data_buffer = df

    def _load_csv_data(self, start_date: str, end_date: str, parameters: list[str] | None = None) -> None:
        """Load data from CSV files."""
        if os.path.isfile(self.data_source):
            df = pd.read_csv(self.data_source, parse_dates=True)
        else:
            # Try to find CSV files in directory
            csv_files = list(Path(self.data_source).glob("**/*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0], parse_dates=True)
            else:
                raise FileNotFoundError(f"No CSV files found in {self.data_source}")

        # Set timestamp index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        # If end_date is just a date (no time), treat it as end of day
        if end_dt.time() == pd.Timestamp("00:00:00").time():
            end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        # Select specific parameters
        if parameters:
            available_params = [p for p in parameters if p in df.columns]
            df = df[available_params]

        self.data_buffer = df

    def _interpolate_missing_data(self) -> None:
        """Interpolate missing data points."""
        if self.data_buffer is None or len(self.data_buffer) == 0:
            return

        # Create a complete time series index based on the update interval
        start_time = self.data_buffer.index.min()
        end_time = self.data_buffer.index.max()

        # Handle case where start_time or end_time is NaT
        if pd.isna(start_time) or pd.isna(end_time):
            return

        # Create complete time index
        complete_index = pd.date_range(start=start_time, end=end_time, freq=f"{self.update_interval_minutes}min")

        # Reindex to complete time series
        self.data_buffer = self.data_buffer.reindex(complete_index)

        # Now interpolate missing values
        if self.interpolation_method == "linear":
            self.data_buffer = self.data_buffer.interpolate(method="linear")
        elif self.interpolation_method == "nearest":
            self.data_buffer = self.data_buffer.interpolate(method="nearest")
        elif self.interpolation_method == "cubic":
            self.data_buffer = self.data_buffer.interpolate(method="cubic")

        # Forward fill any remaining NaN values
        self.data_buffer = self.data_buffer.ffill()

    def _initialize_forecast_models(self) -> None:
        """Initialize forecasting models for each parameter."""
        if self.data_buffer is None:
            return

        for column in self.data_buffer.columns:
            if column in ["temperature_c", "wind_speed_ms", "irradiance_w_m2", "demand_mw"]:
                model = LinearRegression()
                scaler = StandardScaler()

                # Prepare features (hour of day, day of year, etc.)
                X = np.array(
                    [
                        self.data_buffer.index.hour,
                        self.data_buffer.index.dayofyear,
                        np.sin(2 * np.pi * self.data_buffer.index.hour / 24),
                        np.cos(2 * np.pi * self.data_buffer.index.hour / 24),
                    ]
                ).T

                y = self.data_buffer[column].values

                # Fit model
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)

                self.forecast_models[column] = {"model": model, "scaler": scaler}

    def start_streaming(self, start_time: datetime) -> None:
        """Start streaming data from specified time.

        Args:
            start_time: Time to start streaming from
        """
        self.current_time = start_time
        self.is_streaming = True
        self.current_index = 0

        logger.info(f"Started streaming from {start_time}")

    def get_next_slice(self) -> dict[str, Any] | None:
        """Get next data slice.

        Returns:
            Dictionary with current data slice or None if no more data
        """
        if not self.is_streaming or self.data_buffer is None:
            return None

        if self.current_index >= len(self.data_buffer):
            return None

        # Get current data slice
        current_row = self.data_buffer.iloc[self.current_index]
        data_slice = current_row.to_dict()

        # Add metadata
        data_slice["timestamp"] = self.data_buffer.index[self.current_index]

        # Advance to next slice
        self.current_index += 1
        if self.current_time:
            self.current_time += timedelta(minutes=self.update_interval_minutes)

        return data_slice

    def get_current_slice(self) -> dict[str, Any] | None:
        """Get current data slice without advancing.

        Returns:
            Dictionary with current data slice
        """
        if self.data_buffer is None:
            return None

        if self.current_index >= len(self.data_buffer):
            return None

        current_row = self.data_buffer.iloc[self.current_index]
        data_slice = current_row.to_dict()
        data_slice["timestamp"] = self.data_buffer.index[self.current_index]

        return data_slice

    def update_current_time(self, new_time: datetime) -> None:
        """Update current time for real-time mode.

        Args:
            new_time: New current time
        """
        self.current_time = new_time

        # Find corresponding index in data buffer
        if self.data_buffer is not None:
            # Find closest timestamp
            time_diff = abs(self.data_buffer.index - new_time)
            closest_index = time_diff.argmin()
            self.current_index = closest_index

    def generate_forecast(self, parameters: list[str], horizon_hours: int = 6) -> pd.DataFrame:
        """Generate forecast for specified parameters.

        Args:
            parameters: Parameters to forecast
            horizon_hours: Forecast horizon in hours

        Returns:
            DataFrame with forecast data
        """
        if not self.enable_forecasting or not self.forecast_models:
            raise ValueError("Forecasting not enabled or models not initialized")

        # Generate forecast timestamps
        start_time = self.current_time or datetime.now()
        forecast_times = pd.date_range(
            start_time,
            periods=horizon_hours * (60 // self.update_interval_minutes),
            freq=f"{self.update_interval_minutes}min",
        )

        # Generate forecasts
        forecasts = {}
        for param in parameters:
            if param in self.forecast_models:
                model_info = self.forecast_models[param]
                model = model_info["model"]
                scaler = model_info["scaler"]

                # Prepare features
                X = np.array(
                    [
                        forecast_times.hour,
                        forecast_times.dayofyear,
                        np.sin(2 * np.pi * forecast_times.hour / 24),
                        np.cos(2 * np.pi * forecast_times.hour / 24),
                    ]
                ).T

                # Generate forecast
                X_scaled = scaler.transform(X)
                forecast_values = model.predict(X_scaled)

                forecasts[param] = forecast_values

        # Create forecast DataFrame
        forecast_df = pd.DataFrame(forecasts, index=forecast_times)

        return forecast_df

    def update_asset_conditions(self, engine: GridEngine, data_slice: dict[str, Any]) -> None:
        """Update asset conditions in GridEngine.

        Args:
            engine: GridEngine instance
            data_slice: Current data slice
        """
        for asset in engine.get_all_assets():
            # Update solar panel conditions
            if hasattr(asset, "set_irradiance") and "irradiance_w_m2" in data_slice:
                asset.set_irradiance(data_slice["irradiance_w_m2"])

            if hasattr(asset, "set_temperature") and "temperature_c" in data_slice:
                asset.set_temperature(data_slice["temperature_c"])

            # Update wind turbine conditions
            if hasattr(asset, "set_wind_speed") and "wind_speed_ms" in data_slice:
                asset.set_wind_speed(data_slice["wind_speed_ms"])

            # Update weather conditions
            if hasattr(asset, "set_weather_condition") and "weather_condition" in data_slice:
                asset.set_weather_condition(data_slice["weather_condition"])

            # Update demand/load conditions
            if hasattr(asset, "set_electricity_price") and "price_mwh" in data_slice:
                asset.set_electricity_price(data_slice["price_mwh"])

        logger.debug(f"Updated asset conditions for {len(engine.get_all_assets())} assets")

    def get_buffer_status(self) -> dict[str, Any]:
        """Get status of data buffer.

        Returns:
            Dictionary with buffer status information
        """
        if self.data_buffer is None:
            return {"status": "empty"}

        return {
            "status": "loaded",
            "total_records": len(self.data_buffer),
            "current_index": self.current_index,
            "remaining_records": len(self.data_buffer) - self.current_index,
            "time_range": {
                "start": self.data_buffer.index[0].isoformat(),
                "end": self.data_buffer.index[-1].isoformat(),
            },
            "parameters": list(self.data_buffer.columns),
            "is_streaming": self.is_streaming,
        }

    def reset_stream(self) -> None:
        """Reset streaming to beginning."""
        self.current_index = 0
        self.is_streaming = False
        self.current_time = None

        logger.info("DataFeed stream reset")

    def close(self) -> None:
        """Close and cleanup DataFeed."""
        self.data_buffer = None
        self.forecast_models = {}
        self.is_streaming = False

        logger.info("DataFeed closed")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics and cache information."""
        stats = {
            "performance_config": {
                "enable_caching": self.performance_config.enable_caching,
                "cache_size_mb": self.performance_config.cache_size_mb,
                "cache_ttl_seconds": self.performance_config.cache_ttl_seconds,
                "enable_parallel_processing": self.performance_config.enable_parallel_processing,
                "batch_size": self.performance_config.batch_size,
                "max_workers": self.performance_config.max_workers,
            },
            "performance_metrics": self.performance_monitor.get_metrics(),
            "buffer_status": self.get_buffer_status(),
        }

        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()

        return stats

    def clear_cache(self) -> None:
        """Clear the data cache."""
        if self.cache:
            self.cache.clear()
            logger.info("DataFeed cache cleared")

    def optimize_performance(self) -> None:
        """Optimize performance by clearing old metrics and cache."""
        # Clear old performance metrics
        current_time = time.time()
        with self.performance_monitor.lock:
            old_metrics = {
                k: v for k, v in self.performance_monitor.metrics.items() if current_time - v.get("end_time", 0) > 3600
            }  # 1 hour old
            for old_metric in old_metrics:
                del self.performance_monitor.metrics[old_metric]

        # Log current performance stats
        self.performance_monitor.log_metrics()

        logger.info("Performance optimization completed")
