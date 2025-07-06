"""Tests for DataFeed and ETL pipeline functionality in PSIREG simulation system.

This test suite covers:
- Weather data extraction from multiple sources
- ETL pipeline with data transformation and validation
- Parquet storage and efficient retrieval
- Time-series streaming with configurable intervals
- GridEngine integration for asset condition updates
- Error handling and edge cases
"""

import csv
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from psireg.config.schema import GridConfig, SimulationConfig
from psireg.sim.assets.load import Load
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.assets.wind import WindTurbine
from psireg.sim.datafeed import DataFeed, ETLPipeline, WeatherDataExtractor
from psireg.sim.engine import GridEngine
from psireg.utils.enums import AssetStatus


class TestWeatherDataExtractor:
    """Test weather data extraction from various sources."""

    def test_extractor_initialization(self):
        """Test weather data extractor initialization."""
        extractor = WeatherDataExtractor(sources=["nrel", "noaa"], api_key="test_key", cache_dir="/tmp/weather_cache")
        assert extractor.sources == ["nrel", "noaa"]
        assert extractor.api_key == "test_key"
        assert extractor.cache_dir == "/tmp/weather_cache"

    def test_nrel_api_extraction(self):
        """Test NREL API data extraction."""
        extractor = WeatherDataExtractor(sources=["nrel"], api_key="test_key")

        # Mock NREL API response
        mock_response = {
            "outputs": {
                "dnv": [800, 850, 900],  # Direct normal irradiance
                "ghi": [600, 650, 700],  # Global horizontal irradiance
                "dhi": [200, 250, 300],  # Diffuse horizontal irradiance
                "temp_air": [25, 27, 29],  # Air temperature
                "wind_speed": [8, 10, 12],  # Wind speed
                "wind_direction": [180, 190, 200],  # Wind direction
                "pressure": [1013, 1012, 1011],  # Atmospheric pressure
                "relative_humidity": [45, 50, 55],  # Relative humidity
            }
        }

        with patch.object(extractor.pool_manager, "get_session") as mock_session:
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.status_code = 200

            mock_session.return_value.get.return_value = mock_response_obj

            data = extractor.extract_nrel_data(
                latitude=39.7392, longitude=-104.9903, start_date="2024-01-01", end_date="2024-01-02"
            )

            assert len(data) == 3
            assert "irradiance_w_m2" in data.columns
            assert "temperature_c" in data.columns
            assert "wind_speed_ms" in data.columns
            assert "pressure_pa" in data.columns
            assert "humidity_percent" in data.columns

    def test_noaa_api_extraction(self):
        """Test NOAA API data extraction."""
        extractor = WeatherDataExtractor(sources=["noaa"], api_key="test_key")

        # Mock NOAA API response
        mock_response = {
            "data": [
                {
                    "date": "2024-01-01T00:00:00Z",
                    "temperature": 25.5,
                    "wind_speed": 8.2,
                    "humidity": 65,
                    "pressure": 1013.25,
                    "conditions": "Clear",
                },
                {
                    "date": "2024-01-01T01:00:00Z",
                    "temperature": 24.8,
                    "wind_speed": 9.1,
                    "humidity": 68,
                    "pressure": 1012.8,
                    "conditions": "Partly Cloudy",
                },
            ]
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.status_code = 200

            data = extractor.extract_noaa_data(station_id="KDEN", start_date="2024-01-01", end_date="2024-01-02")

            assert len(data) == 2
            assert "temperature_c" in data.columns
            assert "wind_speed_ms" in data.columns
            assert "weather_condition" in data.columns

    def test_local_csv_extraction(self):
        """Test local CSV file data extraction."""
        extractor = WeatherDataExtractor(sources=["csv"])

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "temperature_c", "wind_speed_ms", "irradiance_w_m2"])
            writer.writerow(["2024-01-01T00:00:00Z", 25.0, 8.0, 800])
            writer.writerow(["2024-01-01T01:00:00Z", 26.0, 9.0, 850])
            temp_file = f.name

        try:
            data = extractor.extract_csv_data(temp_file)

            assert len(data) == 2
            assert "temperature_c" in data.columns
            assert "wind_speed_ms" in data.columns
            assert "irradiance_w_m2" in data.columns
            assert isinstance(data.index, pd.DatetimeIndex)

        finally:
            os.unlink(temp_file)

    def test_data_extraction_error_handling(self):
        """Test error handling in data extraction."""
        extractor = WeatherDataExtractor(sources=["nrel"], api_key="invalid_key")

        # Clear cache to ensure we hit the API
        if extractor.cache:
            extractor.cache.clear()

        with patch.object(extractor.pool_manager, "get_session") as mock_session:
            mock_response_obj = Mock()
            mock_response_obj.status_code = 401
            mock_session.return_value.get.return_value = mock_response_obj

            with pytest.raises(Exception) as exc_info:
                extractor.extract_nrel_data(
                    latitude=39.7392, longitude=-104.9903, start_date="2024-01-01", end_date="2024-01-02"
                )

            assert "API request failed" in str(exc_info.value)

    def test_demand_data_extraction(self):
        """Test demand data extraction from various sources."""
        extractor = WeatherDataExtractor(sources=["ercot", "caiso"])

        # Mock ERCOT demand data
        mock_ercot_response = {
            "data": [
                {"timestamp": "2024-01-01T00:00:00Z", "demand_mw": 45000},
                {"timestamp": "2024-01-01T01:00:00Z", "demand_mw": 42000},
            ]
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_ercot_response
            mock_get.return_value.status_code = 200

            data = extractor.extract_demand_data(source="ercot", start_date="2024-01-01", end_date="2024-01-02")

            assert len(data) == 2
            assert "demand_mw" in data.columns
            assert isinstance(data.index, pd.DatetimeIndex)


class TestETLPipeline:
    """Test ETL pipeline functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.parquet_dir = os.path.join(self.test_dir, "parquet")
        os.makedirs(self.parquet_dir, exist_ok=True)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_etl_pipeline_initialization(self):
        """Test ETL pipeline initialization."""
        pipeline = ETLPipeline(
            extractor=WeatherDataExtractor(sources=["nrel"]), storage_path=self.parquet_dir, update_interval_minutes=15
        )

        assert pipeline.storage_path == self.parquet_dir
        assert pipeline.update_interval_minutes == 15
        assert pipeline.extractor is not None

    def test_weather_data_transformation(self):
        """Test weather data transformation and validation."""
        pipeline = ETLPipeline(extractor=WeatherDataExtractor(sources=["nrel"]), storage_path=self.parquet_dir)

        # Create sample raw data
        raw_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="H"),
                "temperature_c": [25 + i * 0.5 for i in range(24)],
                "wind_speed_ms": [8 + i * 0.2 for i in range(24)],
                "irradiance_w_m2": [max(0, 800 * abs(i - 12) / 12) for i in range(24)],
                "humidity_percent": [50 + i for i in range(24)],
            }
        )

        transformed_data = pipeline.transform_weather_data(raw_data)

        assert "weather_condition" in transformed_data.columns
        assert "air_density_kg_m3" in transformed_data.columns
        assert "visibility_km" in transformed_data.columns
        assert transformed_data["temperature_c"].min() >= -50
        assert transformed_data["temperature_c"].max() <= 60
        assert transformed_data["wind_speed_ms"].min() >= 0
        assert transformed_data["irradiance_w_m2"].min() >= 0

    def test_demand_data_transformation(self):
        """Test demand data transformation and validation."""
        pipeline = ETLPipeline(extractor=WeatherDataExtractor(sources=["ercot"]), storage_path=self.parquet_dir)

        # Create sample demand data
        raw_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="H"),
                "demand_mw": [40000 + i * 500 for i in range(24)],
                "price_mwh": [50 + i * 2 for i in range(24)],
            }
        )

        transformed_data = pipeline.transform_demand_data(raw_data)

        assert "demand_category" in transformed_data.columns
        assert "price_level" in transformed_data.columns
        assert transformed_data["demand_mw"].min() >= 0
        assert transformed_data["price_mwh"].min() >= 0

    def test_parquet_storage_and_retrieval(self):
        """Test Parquet storage and retrieval functionality."""
        pipeline = ETLPipeline(extractor=WeatherDataExtractor(sources=["nrel"]), storage_path=self.parquet_dir)

        # Create sample data
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="15min"),
                "temperature_c": [25 + i * 0.1 for i in range(100)],
                "wind_speed_ms": [8 + i * 0.05 for i in range(100)],
                "irradiance_w_m2": [max(0, 800 * abs((i % 96) - 48) / 48) for i in range(100)],
            }
        )

        # Store data
        pipeline.store_parquet_data(data, "weather", partition_cols=["year", "month"])

        # Verify file exists
        parquet_files = list(Path(self.parquet_dir).glob("**/*.parquet"))
        assert len(parquet_files) > 0

        # Retrieve data
        retrieved_data = pipeline.load_parquet_data("weather", start_date="2024-01-01", end_date="2024-01-02")

        assert len(retrieved_data) == 100
        assert "temperature_c" in retrieved_data.columns

    def test_data_quality_validation(self):
        """Test data quality validation and cleaning."""
        pipeline = ETLPipeline(extractor=WeatherDataExtractor(sources=["nrel"]), storage_path=self.parquet_dir)

        # Create data with quality issues
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="H"),
                "temperature_c": [25, -100, 30, None, 35, 200, 40, 45, None, 50],
                "wind_speed_ms": [8, -5, 12, 15, None, 100, 20, 25, 30, 35],
                "irradiance_w_m2": [800, 850, None, 900, 950, -50, 1000, 1050, 2000, 1100],
            }
        )

        cleaned_data = pipeline.validate_and_clean_data(data)

        # Check that outliers and invalid values are handled
        assert cleaned_data["temperature_c"].min() >= -50
        assert cleaned_data["temperature_c"].max() <= 60
        assert cleaned_data["wind_speed_ms"].min() >= 0
        assert cleaned_data["wind_speed_ms"].max() <= 50
        assert cleaned_data["irradiance_w_m2"].min() >= 0
        assert cleaned_data["irradiance_w_m2"].max() <= 1500
        assert cleaned_data.isna().sum().sum() <= len(data) * 0.1  # Max 10% missing

    def test_etl_pipeline_execution(self):
        """Test full ETL pipeline execution."""
        extractor = WeatherDataExtractor(sources=["csv"])
        pipeline = ETLPipeline(extractor=extractor, storage_path=self.parquet_dir)

        # Create sample CSV data
        csv_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="30min"),
                "temperature_c": [25 + i * 0.2 for i in range(48)],
                "wind_speed_ms": [8 + i * 0.1 for i in range(48)],
                "irradiance_w_m2": [max(0, 800 * abs((i % 48) - 24) / 24) for i in range(48)],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # Execute ETL pipeline
            pipeline.execute_etl(source_config={"csv_file": temp_file}, start_date="2024-01-01", end_date="2024-01-02")

            # Verify data was processed and stored
            parquet_files = list(Path(self.parquet_dir).glob("**/*.parquet"))
            assert len(parquet_files) > 0

            # Verify data integrity
            stored_data = pipeline.load_parquet_data("weather", "2024-01-01", "2024-01-02")
            assert len(stored_data) == 48

        finally:
            os.unlink(temp_file)


class TestDataFeed:
    """Test DataFeed streaming functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.parquet_dir = os.path.join(self.test_dir, "parquet")
        os.makedirs(self.parquet_dir, exist_ok=True)

        # Create sample data
        self.sample_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=96, freq="15min"),
                "temperature_c": [25 + 10 * (i % 24) / 24 for i in range(96)],
                "wind_speed_ms": [8 + 5 * (i % 24) / 24 for i in range(96)],
                "irradiance_w_m2": [max(0, 800 * abs((i % 96) - 48) / 48) for i in range(96)],
                "demand_mw": [40000 + 10000 * (i % 24) / 24 for i in range(96)],
            }
        )

        # Store sample data
        table = pa.Table.from_pandas(self.sample_data)
        pq.write_table(table, os.path.join(self.parquet_dir, "sample_data.parquet"))

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_datafeed_initialization(self):
        """Test DataFeed initialization."""
        feed = DataFeed(data_source=self.parquet_dir, update_interval_minutes=15, buffer_size_hours=24)

        assert feed.data_source == self.parquet_dir
        assert feed.update_interval_minutes == 15
        assert feed.buffer_size_hours == 24
        assert feed.current_time is None

    def test_datafeed_data_loading(self):
        """Test DataFeed data loading and buffering."""
        feed = DataFeed(data_source=self.parquet_dir, update_interval_minutes=15, buffer_size_hours=24)

        # Load data
        feed.load_data(start_date="2024-01-01", end_date="2024-01-02")

        assert feed.data_buffer is not None
        assert len(feed.data_buffer) == 96
        assert "temperature_c" in feed.data_buffer.columns

    def test_datafeed_streaming(self):
        """Test DataFeed streaming functionality."""
        feed = DataFeed(data_source=self.parquet_dir, update_interval_minutes=15)

        feed.load_data("2024-01-01", "2024-01-02")
        feed.start_streaming(start_time=datetime(2024, 1, 1))

        # Get first data slice
        data_slice = feed.get_next_slice()
        assert data_slice is not None
        assert "temperature_c" in data_slice
        assert "wind_speed_ms" in data_slice
        assert "irradiance_w_m2" in data_slice

        # Get second data slice
        data_slice2 = feed.get_next_slice()
        assert data_slice2 is not None
        assert data_slice2["temperature_c"] != data_slice["temperature_c"]

    def test_datafeed_interpolation(self):
        """Test DataFeed interpolation for missing data."""
        feed = DataFeed(data_source=self.parquet_dir, update_interval_minutes=15, interpolation_method="linear")

        # Create data with gaps
        sparse_data = self.sample_data.iloc[::3, :].copy()  # Every 3rd row

        # Store sparse data
        table = pa.Table.from_pandas(sparse_data)
        sparse_file = os.path.join(self.parquet_dir, "sparse_data.parquet")
        pq.write_table(table, sparse_file)

        feed.load_data("2024-01-01", "2024-01-02", data_file="sparse_data.parquet")

        # Verify interpolation
        assert len(feed.data_buffer) > len(sparse_data)
        assert feed.data_buffer["temperature_c"].isna().sum() == 0

    def test_datafeed_forecasting(self):
        """Test DataFeed forecasting capability."""
        feed = DataFeed(data_source=self.parquet_dir, update_interval_minutes=15, enable_forecasting=True)

        feed.load_data("2024-01-01", "2024-01-02")

        # Generate forecast
        forecast = feed.generate_forecast(parameters=["temperature_c", "wind_speed_ms"], horizon_hours=6)

        assert len(forecast) == 24  # 6 hours * 4 intervals per hour
        assert "temperature_c" in forecast.columns
        assert "wind_speed_ms" in forecast.columns

    def test_datafeed_real_time_mode(self):
        """Test DataFeed real-time mode."""
        feed = DataFeed(data_source=self.parquet_dir, update_interval_minutes=15, real_time_mode=True)

        feed.load_data("2024-01-01", "2024-01-02")

        # Simulate real-time updates
        current_time = datetime(2024, 1, 1, 12, 0)
        feed.update_current_time(current_time)

        data_slice = feed.get_current_slice()
        assert data_slice is not None

        # Advance time and get next slice
        next_time = current_time + timedelta(minutes=15)
        feed.update_current_time(next_time)

        next_slice = feed.get_current_slice()
        assert next_slice is not None
        assert next_slice["temperature_c"] != data_slice["temperature_c"]


class TestGridEngineIntegration:
    """Test DataFeed integration with GridEngine."""

    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.parquet_dir = os.path.join(self.test_dir, "parquet")
        os.makedirs(self.parquet_dir, exist_ok=True)

        # Create test data
        self.sample_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="30min"),
                "temperature_c": [25 + 10 * (i % 24) / 24 for i in range(48)],
                "wind_speed_ms": [8 + 5 * (i % 24) / 24 for i in range(48)],
                "irradiance_w_m2": [max(0, 800 * abs((i % 48) - 24) / 24) for i in range(48)],
                "demand_mw": [40000 + 10000 * (i % 24) / 24 for i in range(48)],
            }
        )

        # Store test data
        table = pa.Table.from_pandas(self.sample_data)
        pq.write_table(table, os.path.join(self.parquet_dir, "test_data.parquet"))

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_asset_condition_updates(self):
        """Test automatic asset condition updates from DataFeed."""
        # Create GridEngine with assets
        sim_config = SimulationConfig(timestep_minutes=30)
        grid_config = GridConfig()
        engine = GridEngine(sim_config, grid_config)

        # Add test assets
        solar_panel = SolarPanel(
            asset_id="solar_001",
            name="Test Solar Panel",
            node_id="node_1",
            capacity_mw=100.0,
            panel_efficiency=0.20,
            panel_area_m2=50000.0,
        )

        wind_turbine = WindTurbine(
            asset_id="wind_001",
            name="Test Wind Turbine",
            node_id="node_2",
            capacity_mw=50.0,
            rotor_diameter_m=150.0,
            hub_height_m=120.0,
        )

        load_asset = Load(
            asset_id="load_001", name="Test Load", node_id="node_3", capacity_mw=150.0, baseline_demand_mw=100.0
        )

        engine.add_asset(solar_panel)
        engine.add_asset(wind_turbine)
        engine.add_asset(load_asset)

        # Set assets online
        solar_panel.set_status(AssetStatus.ONLINE)
        wind_turbine.set_status(AssetStatus.ONLINE)
        load_asset.set_status(AssetStatus.ONLINE)

        # Create DataFeed
        feed = DataFeed(data_source=self.parquet_dir, update_interval_minutes=30)

        feed.load_data("2024-01-01", "2024-01-02", data_file="test_data.parquet")
        feed.start_streaming(start_time=datetime(2024, 1, 1))

        # Update asset conditions
        data_slice = feed.get_next_slice()
        feed.update_asset_conditions(engine, data_slice)

        # Verify conditions were updated
        assert abs(solar_panel.current_irradiance_w_m2 - data_slice["irradiance_w_m2"]) < 0.1
        assert abs(wind_turbine.current_wind_speed_ms - data_slice["wind_speed_ms"]) < 0.1
        assert abs(solar_panel.current_temperature_c - data_slice["temperature_c"]) < 0.1

    def test_simulation_step_integration(self):
        """Test DataFeed integration with GridEngine simulation steps."""
        # Create GridEngine
        sim_config = SimulationConfig(timestep_minutes=30)
        grid_config = GridConfig()
        engine = GridEngine(sim_config, grid_config)

        # Add assets
        solar_panel = SolarPanel(
            asset_id="solar_001",
            name="Test Solar Panel",
            node_id="node_1",
            capacity_mw=100.0,
            panel_efficiency=0.20,
            panel_area_m2=50000.0,
        )
        engine.add_asset(solar_panel)
        solar_panel.set_status(AssetStatus.ONLINE)

        # Create DataFeed
        feed = DataFeed(data_source=self.parquet_dir, update_interval_minutes=30)

        feed.load_data("2024-01-01", "2024-01-02", data_file="test_data.parquet")
        feed.start_streaming(start_time=datetime(2024, 1, 1))

        # Simulate multiple timesteps
        for _ in range(5):
            # Get data slice
            data_slice = feed.get_next_slice()

            # Update asset conditions
            feed.update_asset_conditions(engine, data_slice)

            # Run simulation step
            engine.step(timedelta(minutes=30))

            # Verify power output changed
            assert solar_panel.current_output_mw >= 0

    def test_datafeed_error_handling(self):
        """Test DataFeed error handling and recovery."""
        feed = DataFeed(data_source="/nonexistent/path", update_interval_minutes=15)

        # Test handling of missing data source
        with pytest.raises(FileNotFoundError):
            feed.load_data("2024-01-01", "2024-01-02")

        # Test handling of invalid date ranges
        feed.data_source = self.parquet_dir
        with pytest.raises(ValueError):
            feed.load_data("2024-01-02", "2024-01-01")  # End before start

        # Test handling of empty data
        empty_data = pd.DataFrame(columns=["timestamp", "temperature_c"])
        table = pa.Table.from_pandas(empty_data)
        pq.write_table(table, os.path.join(self.parquet_dir, "empty_data.parquet"))

        feed.load_data("2024-01-01", "2024-01-02", data_file="empty_data.parquet")

        # Should handle gracefully
        assert feed.data_buffer is not None
        assert len(feed.data_buffer) == 0

    def test_datafeed_performance_optimization(self):
        """Test DataFeed performance optimization features."""
        # Create large dataset
        large_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=8760, freq="H"),  # 1 year
                "temperature_c": [25 + 10 * (i % 8760) / 8760 for i in range(8760)],
                "wind_speed_ms": [8 + 5 * (i % 8760) / 8760 for i in range(8760)],
                "irradiance_w_m2": [max(0, 800 * abs((i % 8760) - 4380) / 4380) for i in range(8760)],
            }
        )

        # Store with partitioning
        table = pa.Table.from_pandas(large_data)
        pq.write_table(table, os.path.join(self.parquet_dir, "large_data.parquet"))

        # Test chunked loading
        feed = DataFeed(data_source=self.parquet_dir, update_interval_minutes=60, chunk_size_hours=24)

        feed.load_data("2024-01-01", "2024-01-02", data_file="large_data.parquet")

        # Should only load requested timeframe
        assert len(feed.data_buffer) == 48  # 2 days of hourly data (2024-01-01 to 2024-01-02)

    def test_reusable_feed_interface(self):
        """Test reusable DataFeed interface with multiple configurations."""
        # Configuration 1: Weather-focused
        weather_config = {
            "data_source": self.parquet_dir,
            "update_interval_minutes": 15,
            "parameters": ["temperature_c", "wind_speed_ms", "irradiance_w_m2"],
            "interpolation_method": "linear",
        }

        weather_feed = DataFeed(**weather_config)
        weather_feed.load_data("2024-01-01", "2024-01-02", data_file="test_data.parquet")
        weather_feed.start_streaming(start_time=datetime(2024, 1, 1))

        # Configuration 2: Demand-focused
        demand_config = {
            "data_source": self.parquet_dir,
            "update_interval_minutes": 30,
            "parameters": ["demand_mw"],
            "interpolation_method": "nearest",
        }

        demand_feed = DataFeed(**demand_config)
        demand_feed.load_data("2024-01-01", "2024-01-02", data_file="test_data.parquet")
        demand_feed.start_streaming(start_time=datetime(2024, 1, 1))

        # Both feeds should work independently
        weather_slice = weather_feed.get_next_slice()
        demand_slice = demand_feed.get_next_slice()

        # Weather feed should have weather parameters
        assert "temperature_c" in weather_slice
        assert "wind_speed_ms" in weather_slice
        assert "irradiance_w_m2" in weather_slice

        # Demand feed should have demand parameters
        assert "demand_mw" in demand_slice

        # Both feeds should have the same timestamp values since they're reading from the same source
        assert weather_slice["timestamp"] == demand_slice["timestamp"]

        # Verify feed independence: weather feed should have weather params, demand feed should have all params
        assert "temperature_c" in weather_slice
        assert "temperature_c" in demand_slice  # demand feed loads all parameters by default
