"""Tests for MetricsCollector hooks and time-series logging functionality."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from psireg.sim.engine import GridEngine
from psireg.sim.metrics import MAECalculator, MetricsCollector, MetricsHook


class TestMetricsCollectorHooks:
    """Test MetricsCollector hooks system."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def metrics_collector(self, temp_log_dir):
        """Create MetricsCollector instance for testing."""
        return MetricsCollector(
            log_directory=temp_log_dir,
            collection_interval_seconds=60,
            enable_time_series_logging=True,
            enable_mae_calculation=True,
        )

    @pytest.fixture
    def mock_grid_engine(self):
        """Create mock GridEngine for testing."""
        engine = Mock(spec=GridEngine)
        engine.current_time = datetime(2024, 1, 1, 12, 0, 0)
        engine.get_state.return_value = Mock(
            frequency_hz=60.05,
            total_generation_mw=100.0,
            total_load_mw=95.0,
            total_storage_mw=5.0,
            power_balance_mw=10.0,
        )
        engine.get_all_assets.return_value = []
        engine.nodes = {}
        return engine

    def test_metrics_collector_initialization(self, temp_log_dir):
        """Test MetricsCollector initialization with various configurations."""
        # Test basic initialization
        collector = MetricsCollector(log_directory=temp_log_dir, collection_interval_seconds=30)

        assert collector.log_directory == temp_log_dir
        assert collector.collection_interval_seconds == 30
        assert collector.enable_time_series_logging is True
        assert collector.enable_mae_calculation is False
        assert len(collector.hooks) == 0
        assert collector.time_series_data == []
        assert collector.mae_calculator is None

    def test_metrics_collector_with_mae_enabled(self, temp_log_dir):
        """Test MetricsCollector initialization with MAE calculation enabled."""
        collector = MetricsCollector(
            log_directory=temp_log_dir, collection_interval_seconds=60, enable_mae_calculation=True
        )

        assert collector.mae_calculator is not None
        assert isinstance(collector.mae_calculator, MAECalculator)

    def test_hook_registration(self, metrics_collector):
        """Test hook registration and management."""
        # Create custom hook
        hook = MetricsHook(
            name="test_hook", collect_function=lambda engine, collector: {"test_metric": 42.0}, frequency_seconds=30
        )

        # Register hook
        metrics_collector.register_hook(hook)

        assert len(metrics_collector.hooks) == 1
        assert metrics_collector.hooks[0] == hook
        assert hook.name == "test_hook"
        assert hook.frequency_seconds == 30

    def test_hook_execution(self, metrics_collector, mock_grid_engine):
        """Test hook execution and data collection."""
        # Create test hook
        test_data = {"custom_metric": 123.45}
        hook = MetricsHook(name="test_hook", collect_function=lambda engine, collector: test_data, frequency_seconds=60)

        metrics_collector.register_hook(hook)

        # Execute hooks
        collected_data = metrics_collector._execute_hooks(mock_grid_engine)

        assert "test_hook" in collected_data
        assert collected_data["test_hook"] == test_data

    def test_built_in_hooks_registration(self, metrics_collector):
        """Test built-in hooks are registered correctly."""
        metrics_collector.register_default_hooks()

        hook_names = [hook.name for hook in metrics_collector.hooks]
        expected_hooks = [
            "power_generation_metrics",
            "curtailment_metrics",
            "frequency_deviation_metrics",
            "voltage_deviation_metrics",
            "fossil_fuel_metrics",
        ]

        for expected_hook in expected_hooks:
            assert expected_hook in hook_names

    def test_time_series_data_collection(self, metrics_collector, mock_grid_engine):
        """Test time-series data collection and storage."""
        # Register default hooks
        metrics_collector.register_default_hooks()

        # Collect metrics
        metrics_collector.collect_metrics(mock_grid_engine)

        # Verify data was collected
        assert len(metrics_collector.time_series_data) == 1

        collected_data = metrics_collector.time_series_data[0]
        assert "timestamp" in collected_data
        assert "power_generation_metrics" in collected_data
        assert "curtailment_metrics" in collected_data
        assert "frequency_deviation_metrics" in collected_data

    def test_collection_interval_filtering(self, metrics_collector, mock_grid_engine):
        """Test that hooks are only executed based on their frequency."""
        # Hook with 30-second frequency
        fast_hook = MetricsHook(
            name="fast_hook", collect_function=lambda engine, collector: {"fast_metric": 1.0}, frequency_seconds=30
        )

        # Hook with 120-second frequency
        slow_hook = MetricsHook(
            name="slow_hook", collect_function=lambda engine, collector: {"slow_metric": 2.0}, frequency_seconds=120
        )

        metrics_collector.register_hook(fast_hook)
        metrics_collector.register_hook(slow_hook)

        # First collection (both should execute)
        metrics_collector.collect_metrics(mock_grid_engine)
        data1 = metrics_collector.time_series_data[0]

        assert "fast_hook" in data1
        assert "slow_hook" in data1

        # Simulate 60 seconds later
        mock_grid_engine.current_time = datetime(2024, 1, 1, 12, 1, 0)

        # Second collection (only fast hook should execute)
        metrics_collector.collect_metrics(mock_grid_engine)
        data2 = metrics_collector.time_series_data[1]

        assert "fast_hook" in data2
        assert "slow_hook" not in data2

    def test_hook_error_handling(self, metrics_collector, mock_grid_engine):
        """Test error handling in hook execution."""

        # Create hook that raises exception
        def failing_hook(engine, collector):
            raise ValueError("Test error")

        hook = MetricsHook(name="failing_hook", collect_function=failing_hook, frequency_seconds=60)

        metrics_collector.register_hook(hook)

        # Hook should not prevent other hooks from executing
        success_hook = MetricsHook(
            name="success_hook", collect_function=lambda engine, collector: {"success": True}, frequency_seconds=60
        )

        metrics_collector.register_hook(success_hook)

        # Collect metrics (should not raise exception)
        metrics_collector.collect_metrics(mock_grid_engine)

        # Verify successful hook still executed
        data = metrics_collector.time_series_data[0]
        assert "success_hook" in data
        assert data["success_hook"]["success"] is True

    def test_csv_export(self, metrics_collector, mock_grid_engine, temp_log_dir):
        """Test CSV export functionality."""
        # Register hooks and collect some data
        metrics_collector.register_default_hooks()

        # Collect multiple data points
        for i in range(3):
            mock_grid_engine.current_time = datetime(2024, 1, 1, 12, i, 0)
            metrics_collector.collect_metrics(mock_grid_engine)

        # Export to CSV
        csv_path = metrics_collector.export_to_csv()

        assert csv_path.exists()
        assert csv_path.suffix == ".csv"

        # Verify CSV content
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert "timestamp" in df.columns

    def test_json_export(self, metrics_collector, mock_grid_engine, temp_log_dir):
        """Test JSON export functionality."""
        # Register hooks and collect some data
        metrics_collector.register_default_hooks()
        metrics_collector.collect_metrics(mock_grid_engine)

        # Export to JSON
        json_path = metrics_collector.export_to_json()

        assert json_path.exists()
        assert json_path.suffix == ".json"

        # Verify JSON content
        with open(json_path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert "timestamp" in data[0]

    def test_metrics_collector_reset(self, metrics_collector, mock_grid_engine):
        """Test metrics collector reset functionality."""
        # Collect some data
        metrics_collector.register_default_hooks()
        metrics_collector.collect_metrics(mock_grid_engine)

        assert len(metrics_collector.time_series_data) == 1

        # Reset collector
        metrics_collector.reset()

        assert len(metrics_collector.time_series_data) == 0
        assert len(metrics_collector.hooks) == 0

    def test_get_summary_statistics(self, metrics_collector, mock_grid_engine):
        """Test summary statistics generation."""
        # Register hooks and collect data
        metrics_collector.register_default_hooks()

        # Collect multiple data points with varying values
        for i in range(5):
            mock_grid_engine.current_time = datetime(2024, 1, 1, 12, i, 0)
            mock_grid_engine.get_state.return_value.frequency_hz = 60.0 + i * 0.01
            metrics_collector.collect_metrics(mock_grid_engine)

        # Get summary statistics
        summary = metrics_collector.get_summary_statistics()

        assert isinstance(summary, dict)
        assert "data_points" in summary
        assert summary["data_points"] == 5
        assert "time_range" in summary
        assert "frequency_deviation_stats" in summary


class TestMetricsHookClass:
    """Test MetricsHook class functionality."""

    def test_hook_initialization(self):
        """Test MetricsHook initialization."""

        def collect_func(engine, collector):
            return {"test": 1.0}

        hook = MetricsHook(
            name="test_hook",
            collect_function=collect_func,
            frequency_seconds=60,
            description="Test hook for unit tests",
        )

        assert hook.name == "test_hook"
        assert hook.collect_function == collect_func
        assert hook.frequency_seconds == 60
        assert hook.description == "Test hook for unit tests"
        assert hook.last_execution_time is None

    def test_hook_should_execute(self):
        """Test hook execution timing logic."""
        hook = MetricsHook(name="test_hook", collect_function=lambda engine, collector: {}, frequency_seconds=60)

        current_time = datetime(2024, 1, 1, 12, 0, 0)

        # First execution should always run
        assert hook.should_execute(current_time) is True

        # Mark as executed
        hook.last_execution_time = current_time

        # Should not execute again immediately
        assert hook.should_execute(current_time) is False

        # Should execute after frequency interval
        later_time = current_time + timedelta(seconds=60)
        assert hook.should_execute(later_time) is True

    def test_hook_execution_with_error(self):
        """Test hook execution with error handling."""

        def failing_function(engine, collector):
            raise RuntimeError("Test error")

        hook = MetricsHook(name="failing_hook", collect_function=failing_function, frequency_seconds=60)

        mock_engine = Mock()
        mock_collector = Mock()

        # Execute hook (should handle error gracefully)
        result = hook.execute(mock_engine, mock_collector)

        # Should return None on error
        assert result is None


class TestTimeSeriesLogging:
    """Test time-series logging functionality."""

    def test_time_series_data_structure(self):
        """Test time-series data structure and consistency."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            collector = MetricsCollector(log_directory=Path(tmp_dir), enable_time_series_logging=True)

            # Mock data
            test_data = {
                "timestamp": datetime(2024, 1, 1, 12, 0, 0),
                "power_metrics": {"generation_mw": 100.0, "load_mw": 95.0},
                "frequency_metrics": {"frequency_hz": 60.05, "deviation_hz": 0.05},
            }

            collector.time_series_data.append(test_data)

            # Verify data structure
            assert len(collector.time_series_data) == 1
            data_point = collector.time_series_data[0]

            assert "timestamp" in data_point
            assert "power_metrics" in data_point
            assert "frequency_metrics" in data_point

            assert isinstance(data_point["timestamp"], datetime)
            assert isinstance(data_point["power_metrics"], dict)

    def test_continuous_logging_performance(self):
        """Test performance of continuous logging."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            collector = MetricsCollector(log_directory=Path(tmp_dir), enable_time_series_logging=True)

            # Simulate continuous logging
            start_time = datetime(2024, 1, 1, 12, 0, 0)

            for i in range(100):
                data_point = {"timestamp": start_time + timedelta(seconds=i * 60), "metrics": {"value": i * 1.5}}
                collector.time_series_data.append(data_point)

            # Verify performance
            assert len(collector.time_series_data) == 100

            # Test export performance
            csv_path = collector.export_to_csv()
            assert csv_path.exists()

    def test_log_file_naming_convention(self):
        """Test log file naming convention."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            collector = MetricsCollector(log_directory=Path(tmp_dir), enable_time_series_logging=True)

            # Add some test data
            collector.time_series_data.append({"timestamp": datetime(2024, 1, 1, 12, 0, 0), "test": "data"})

            # Export files
            csv_path = collector.export_to_csv()
            json_path = collector.export_to_json()

            # Verify naming convention
            assert "metrics_" in csv_path.name
            assert "metrics_" in json_path.name
            assert csv_path.name.endswith(".csv")
            assert json_path.name.endswith(".json")

            # Verify timestamp pattern in filename (current year)
        import re

        timestamp_pattern = r"\d{8}_\d{6}"  # YYYYMMDD_HHMMSS
        assert re.search(timestamp_pattern, csv_path.name) is not None
