"""
Integration tests for the DataFeed system.

These tests verify the entire pipeline from data extraction to streaming,
including performance optimizations, error handling, and real-world scenarios.
"""

import os
import shutil
import tempfile
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from psireg.sim.assets.battery import Battery
from psireg.sim.assets.load import Load
from psireg.sim.assets.solar import SolarPanel
from psireg.sim.assets.wind import WindTurbine
from psireg.sim.datafeed import (
    ConnectionPoolManager,
    DataCache,
    DataFeed,
    ETLPipeline,
    PerformanceConfig,
    PerformanceMonitor,
    WeatherDataExtractor,
)
from psireg.sim.engine import GridEngine


class TestDataFeedIntegration:
    """Integration tests for the complete DataFeed pipeline."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_dir = os.path.join(self.temp_dir, "parquet")
        self.csv_dir = os.path.join(self.temp_dir, "csv")
        os.makedirs(self.parquet_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        # Create performance config
        self.performance_config = PerformanceConfig(
            enable_caching=True,
            cache_size_mb=50,
            cache_ttl_seconds=1800,
            enable_parallel_processing=True,
            batch_size=500,
            max_workers=2,
        )

        # Create test data
        self.create_test_data()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_data(self):
        """Create comprehensive test data for integration tests."""
        # Create weather data
        dates = pd.date_range("2024-01-01", periods=168, freq="h")  # 1 week
        weather_conditions = ["CLEAR", "PARTLY_CLOUDY", "CLOUDY", "RAINY", "WINDY"]

        weather_data = pd.DataFrame(
            {
                "timestamp": dates,
                "temperature_c": [
                    20 + 10 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 2) for i in range(len(dates))
                ],
                "wind_speed_ms": [
                    10 + 5 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 1) for i in range(len(dates))
                ],
                "irradiance_w_m2": [
                    max(0, 800 * np.sin(np.pi * (i % 24) / 24)) + np.random.normal(0, 50) for i in range(len(dates))
                ],
                "humidity_percent": [
                    60 + 20 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 5) for i in range(len(dates))
                ],
                "pressure_hpa": [
                    1013 + 10 * np.sin(2 * np.pi * i / 168) + np.random.normal(0, 2) for i in range(len(dates))
                ],
                "weather_condition": [weather_conditions[i % len(weather_conditions)] for i in range(len(dates))],
                "air_density_kg_m3": [1.225 + np.random.normal(0, 0.05) for _ in range(len(dates))],
                "visibility_km": [15 + np.random.normal(0, 3) for _ in range(len(dates))],
            }
        )

        # Create demand data
        demand_categories = ["LOW", "MEDIUM", "HIGH", "PEAK"]
        price_levels = ["LOW", "MEDIUM", "HIGH"]

        demand_data = pd.DataFrame(
            {
                "timestamp": dates,
                "demand_mw": [
                    15000 + 5000 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 1000) for i in range(len(dates))
                ],
                "price_mwh": [50 + 30 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 5) for i in range(len(dates))],
                "demand_category": [demand_categories[i % len(demand_categories)] for i in range(len(dates))],
                "price_level": [price_levels[i % len(price_levels)] for i in range(len(dates))],
            }
        )

        # Save as CSV
        weather_data.to_csv(os.path.join(self.csv_dir, "weather.csv"), index=False)
        demand_data.to_csv(os.path.join(self.csv_dir, "demand.csv"), index=False)

        # Save as Parquet if available
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq  # noqa: F401

            weather_data.set_index("timestamp").to_parquet(os.path.join(self.parquet_dir, "weather.parquet"))
            demand_data.set_index("timestamp").to_parquet(os.path.join(self.parquet_dir, "demand.parquet"))
        except ImportError:
            pass

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline from ETL to streaming."""
        # Initialize extractor
        extractor = WeatherDataExtractor(sources=["csv"], performance_config=self.performance_config)

        # Initialize ETL pipeline
        etl = ETLPipeline(extractor=extractor, storage_path=self.parquet_dir)

        # Execute ETL
        source_config = {"csv_file": os.path.join(self.csv_dir, "weather.csv")}
        etl.execute_etl(source_config, "2024-01-01", "2024-01-07", ["weather"])

        # Initialize DataFeed
        feed = DataFeed(
            data_source=self.parquet_dir, performance_config=self.performance_config, enable_forecasting=True
        )

        # Load data
        feed.load_data("2024-01-01", "2024-01-07")

        # Start streaming
        feed.start_streaming(datetime(2024, 1, 1, 0, 0, 0))

        # Test streaming
        slices = []
        for _ in range(10):
            data_slice = feed.get_next_slice()
            assert data_slice is not None
            slices.append(data_slice)

        # Verify data integrity
        assert len(slices) == 10
        for slice_data in slices:
            assert "timestamp" in slice_data
            assert "temperature_c" in slice_data
            assert "wind_speed_ms" in slice_data
            assert "irradiance_w_m2" in slice_data

        # Test performance stats
        stats = feed.get_performance_stats()
        assert "performance_config" in stats
        assert "performance_metrics" in stats
        assert "cache_stats" in stats
        assert "buffer_status" in stats

        # Test forecasting
        forecast = feed.generate_forecast(["temperature_c", "wind_speed_ms"], horizon_hours=24)
        assert len(forecast) > 0
        assert "temperature_c" in forecast.columns
        assert "wind_speed_ms" in forecast.columns

        feed.close()

    def test_performance_optimization(self):
        """Test performance optimization features."""
        # Test caching
        cache = DataCache(max_size_mb=10, ttl_seconds=60)

        # Test cache operations
        test_data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cache.put("test_key", test_data)

        retrieved_data = cache.get("test_key")
        assert retrieved_data is not None
        assert retrieved_data.equals(test_data)

        # Test cache stats
        stats = cache.get_stats()
        assert stats["cache_entries"] == 1
        assert stats["cache_size_mb"] > 0

        # Test connection pool
        pool_manager = ConnectionPoolManager(pool_size=5)
        session1 = pool_manager.get_session("https://example.com")
        session2 = pool_manager.get_session("https://example.com")
        assert session1 is session2  # Should reuse session

        # Test performance monitor
        monitor = PerformanceMonitor()
        timer_id = monitor.start_timer("test_operation")
        time.sleep(0.1)
        duration = monitor.end_timer(timer_id)
        assert duration >= 0.1

        metrics = monitor.get_metrics()
        assert timer_id in metrics
        assert metrics[timer_id]["operation"] == "test_operation"
        assert metrics[timer_id]["status"] == "completed"

        pool_manager.close_all()

    def test_grid_engine_integration(self):
        """Test integration with GridEngine."""
        # Import required types
        from psireg.config.schema import GridConfig, SimulationConfig
        from psireg.sim.engine import NetworkNode
        from psireg.utils.enums import AssetStatus

        # Create GridEngine with assets
        engine = GridEngine(simulation_config=SimulationConfig(), grid_config=GridConfig())

        # Add network nodes
        gen_node = NetworkNode(node_id="gen_node", name="Generation Node", voltage_kv=138.0)
        storage_node = NetworkNode(node_id="storage_node", name="Storage Node", voltage_kv=138.0)
        load_node = NetworkNode(node_id="load_node", name="Load Node", voltage_kv=138.0)

        engine.add_node(gen_node)
        engine.add_node(storage_node)
        engine.add_node(load_node)

        # Add various assets
        solar = SolarPanel(
            asset_id="solar1",
            name="Solar Panel 1",
            node_id="gen_node",
            capacity_mw=100,
            panel_efficiency=0.2,
            panel_area_m2=50000.0,
        )
        wind = WindTurbine(
            asset_id="wind1",
            name="Wind Turbine 1",
            node_id="gen_node",
            capacity_mw=50,
            cut_in_speed_ms=3,
            cut_out_speed_ms=25,
            rotor_diameter_m=80.0,
        )
        battery = Battery(
            asset_id="battery1", name="Battery 1", node_id="storage_node", capacity_mw=50, energy_capacity_mwh=200
        )
        load = Load(asset_id="load1", name="Load 1", node_id="load_node", capacity_mw=75, baseline_demand_mw=60)

        engine.add_asset(solar)
        engine.add_asset(wind)
        engine.add_asset(battery)
        engine.add_asset(load)

        # Set assets to online status
        solar.set_status(AssetStatus.ONLINE)
        wind.set_status(AssetStatus.ONLINE)
        battery.set_status(AssetStatus.ONLINE)
        load.set_status(AssetStatus.ONLINE)

        # Initialize DataFeed
        weather_csv_path = os.path.join(self.csv_dir, "weather.csv")
        feed = DataFeed(data_source=weather_csv_path, performance_config=self.performance_config)

        # Load weather data
        feed.load_data("2024-01-01", "2024-01-02")
        feed.start_streaming(datetime(2024, 1, 1, 0, 0, 0))

        # Test integration
        engine.get_state()

        # Update asset conditions using DataFeed
        for _ in range(5):
            data_slice = feed.get_next_slice()
            if data_slice:
                feed.update_asset_conditions(engine, data_slice)

                # Verify asset conditions were updated
                current_state = engine.get_state()
                assert current_state is not None
                assert current_state.total_generation_mw >= 0
                assert current_state.total_load_mw >= 0

        feed.close()

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with invalid data source
        with pytest.raises(FileNotFoundError):
            feed = DataFeed(data_source="/nonexistent/path")
            feed.load_data("2024-01-01", "2024-01-02")

        # Test with invalid date range
        feed = DataFeed(data_source=self.csv_dir)
        with pytest.raises(ValueError):
            feed.load_data("2024-01-02", "2024-01-01")  # End before start

        # Test graceful degradation
        weather_csv_path = os.path.join(self.csv_dir, "weather.csv")
        feed = DataFeed(data_source=weather_csv_path, performance_config=PerformanceConfig(enable_caching=False))
        feed.load_data("2024-01-01", "2024-01-02")

        # Should work without caching
        feed.start_streaming(datetime(2024, 1, 1, 0, 0, 0))
        data_slice = feed.get_next_slice()
        assert data_slice is not None

        feed.close()

    def test_scalability_and_performance(self):
        """Test scalability and performance under load."""
        # Create multiple DataFeeds
        feeds = []
        weather_csv_path = os.path.join(self.csv_dir, "weather.csv")

        for _ in range(3):
            config = PerformanceConfig(enable_caching=True, cache_size_mb=20, batch_size=100, max_workers=2)
            feed = DataFeed(data_source=weather_csv_path, performance_config=config)
            feeds.append(feed)

        # Load data concurrently
        start_time = time.time()

        for feed in feeds:
            feed.load_data("2024-01-01", "2024-01-07")

        load_time = time.time() - start_time

        # Test streaming performance
        start_time = time.time()

        for feed in feeds:
            feed.start_streaming(datetime(2024, 1, 1, 0, 0, 0))

            # Get multiple slices
            for _ in range(10):
                data_slice = feed.get_next_slice()
                assert data_slice is not None

        streaming_time = time.time() - start_time

        # Verify performance (reasonable thresholds)
        assert load_time < 30.0  # Should load within 30 seconds
        assert streaming_time < 10.0  # Should stream within 10 seconds

        # Test performance stats
        for feed in feeds:
            stats = feed.get_performance_stats()
            assert "performance_metrics" in stats
            assert "cache_stats" in stats
            feed.close()

        # Verify cleanup
        for feed in feeds:
            assert feed.data_buffer is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
