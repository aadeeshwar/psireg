"""Tests for curtailment %, Hz/V deviation metrics collection from existing infrastructure."""

from datetime import datetime
from unittest.mock import Mock

import numpy as np
from psireg.sim.assets.base import AssetType
from psireg.sim.metrics import (
    CurtailmentCalculator,
    FrequencyDeviationCalculator,
    MetricsCollector,
    VoltageDeviationCalculator,
)


class TestCurtailmentMetrics:
    """Test curtailment percentage calculation and tracking."""

    def test_curtailment_calculator_initialization(self):
        """Test CurtailmentCalculator initialization."""
        calculator = CurtailmentCalculator()

        assert calculator.renewable_types == [AssetType.SOLAR, AssetType.WIND, AssetType.HYDRO]
        assert calculator.curtailment_threshold_mw == 0.001  # Minimum threshold for curtailment detection

    def test_calculate_curtailment_percentage_no_curtailment(self):
        """Test curtailment calculation with no curtailment."""
        calculator = CurtailmentCalculator()

        # Create mock renewable assets at full output
        solar_asset = Mock()
        solar_asset.asset_type = AssetType.SOLAR
        solar_asset.current_output_mw = 100.0
        solar_asset.capacity_mw = 100.0
        solar_asset.is_online.return_value = True
        solar_asset.get_theoretical_max_power.return_value = 100.0
        solar_asset.curtailment_factor = 1.0

        wind_asset = Mock()
        wind_asset.asset_type = AssetType.WIND
        wind_asset.current_output_mw = 80.0
        wind_asset.capacity_mw = 100.0
        wind_asset.is_online.return_value = True
        wind_asset.get_theoretical_max_power.return_value = 80.0
        wind_asset.curtailment_factor = 1.0

        assets = [solar_asset, wind_asset]

        result = calculator.calculate_curtailment_metrics(assets)

        assert result["curtailment_percentage"] == 0.0
        assert result["total_renewable_generation_mw"] == 180.0
        assert result["potential_renewable_generation_mw"] == 180.0
        assert result["curtailed_energy_mw"] == 0.0

    def test_calculate_curtailment_percentage_with_curtailment(self):
        """Test curtailment calculation with active curtailment."""
        calculator = CurtailmentCalculator()

        # Create mock renewable assets with curtailment
        solar_asset = Mock()
        solar_asset.asset_type = AssetType.SOLAR
        solar_asset.current_output_mw = 80.0  # Curtailed from 100 MW
        solar_asset.capacity_mw = 100.0
        solar_asset.is_online.return_value = True
        solar_asset.get_theoretical_max_power.return_value = 100.0
        solar_asset.curtailment_factor = 0.8  # 20% curtailment

        wind_asset = Mock()
        wind_asset.asset_type = AssetType.WIND
        wind_asset.current_output_mw = 60.0  # Curtailed from 80 MW
        wind_asset.capacity_mw = 100.0
        wind_asset.is_online.return_value = True
        wind_asset.get_theoretical_max_power.return_value = 80.0
        wind_asset.curtailment_factor = 0.75  # 25% curtailment

        assets = [solar_asset, wind_asset]

        result = calculator.calculate_curtailment_metrics(assets)

        # Total actual: 140 MW, Total potential: 180 MW, Curtailed: 40 MW
        assert abs(result["curtailment_percentage"] - 22.22) < 0.01  # 40/180 * 100
        assert result["total_renewable_generation_mw"] == 140.0
        assert result["potential_renewable_generation_mw"] == 180.0
        assert result["curtailed_energy_mw"] == 40.0

    def test_calculate_curtailment_by_asset_type(self):
        """Test curtailment calculation broken down by asset type."""
        calculator = CurtailmentCalculator()

        # Create solar assets with different curtailment levels
        solar1 = Mock()
        solar1.asset_type = AssetType.SOLAR
        solar1.current_output_mw = 90.0
        solar1.get_theoretical_max_power.return_value = 100.0
        solar1.is_online.return_value = True

        solar2 = Mock()
        solar2.asset_type = AssetType.SOLAR
        solar2.current_output_mw = 80.0
        solar2.get_theoretical_max_power.return_value = 100.0
        solar2.is_online.return_value = True

        # Create wind assets
        wind1 = Mock()
        wind1.asset_type = AssetType.WIND
        wind1.current_output_mw = 70.0
        wind1.get_theoretical_max_power.return_value = 80.0
        wind1.is_online.return_value = True

        assets = [solar1, solar2, wind1]

        result = calculator.calculate_curtailment_by_type(assets)

        assert result["solar_curtailment_mw"] == 30.0  # (100-90) + (100-80)
        assert result["solar_potential_mw"] == 200.0
        assert result["solar_curtailment_percentage"] == 15.0  # 30/200 * 100

        assert result["wind_curtailment_mw"] == 10.0  # 80-70
        assert result["wind_potential_mw"] == 80.0
        assert result["wind_curtailment_percentage"] == 12.5  # 10/80 * 100

    def test_calculate_economic_curtailment_impact(self):
        """Test calculation of economic impact of curtailment."""
        calculator = CurtailmentCalculator()

        # Create mock renewable assets with curtailment
        solar_asset = Mock()
        solar_asset.asset_type = AssetType.SOLAR
        solar_asset.current_output_mw = 80.0
        solar_asset.get_theoretical_max_power.return_value = 100.0
        solar_asset.is_online.return_value = True
        solar_asset.calculate_revenue_loss_per_hour.return_value = 800.0  # $800/hr lost

        assets = [solar_asset]
        electricity_price_mwh = 40.0  # $/MWh

        result = calculator.calculate_economic_impact(assets, electricity_price_mwh)

        # 20 MW curtailed * $40/MWh = $800/hr
        assert result["curtailed_energy_mw"] == 20.0
        assert result["revenue_loss_per_hour"] == 800.0
        assert result["electricity_price_mwh"] == 40.0

    def test_curtailment_time_series_tracking(self):
        """Test time-series tracking of curtailment events."""
        calculator = CurtailmentCalculator()

        # Simulate multiple time points
        time_series_data = []

        for hour in range(6):  # 6 hours of data
            # Simulate varying curtailment levels
            curtailment_factor = 0.8 + 0.1 * np.sin(hour * np.pi / 3)  # Varies 0.7-0.9

            solar_asset = Mock()
            solar_asset.asset_type = AssetType.SOLAR
            solar_asset.current_output_mw = 100.0 * curtailment_factor
            solar_asset.get_theoretical_max_power.return_value = 100.0
            solar_asset.is_online.return_value = True

            assets = [solar_asset]
            timestamp = datetime(2024, 1, 1, 12 + hour, 0, 0)

            metrics = calculator.calculate_curtailment_metrics(assets)
            metrics["timestamp"] = timestamp

            time_series_data.append(metrics)

        # Verify time series data
        assert len(time_series_data) == 6

        # Check that curtailment varies over time
        curtailment_values = [data["curtailment_percentage"] for data in time_series_data]
        assert max(curtailment_values) > min(curtailment_values)


class TestFrequencyDeviationMetrics:
    """Test frequency deviation calculation and tracking."""

    def test_frequency_deviation_calculator_initialization(self):
        """Test FrequencyDeviationCalculator initialization."""
        calculator = FrequencyDeviationCalculator(
            nominal_frequency_hz=60.0, alert_threshold_hz=0.1, emergency_threshold_hz=0.5
        )

        assert calculator.nominal_frequency_hz == 60.0
        assert calculator.alert_threshold_hz == 0.1
        assert calculator.emergency_threshold_hz == 0.5

    def test_calculate_frequency_deviation_normal(self):
        """Test frequency deviation calculation under normal conditions."""
        calculator = FrequencyDeviationCalculator()

        # Mock grid engine with normal frequency
        mock_engine = Mock()
        mock_engine.get_state.return_value.frequency_hz = 60.02

        result = calculator.calculate_frequency_metrics(mock_engine)

        assert result["current_frequency_hz"] == 60.02
        assert result["frequency_deviation_hz"] == 0.02
        assert abs(result["frequency_deviation_percentage"] - 0.0333) < 0.001  # 0.02/60 * 100
        assert result["frequency_status"] == "normal"
        assert result["is_emergency"] is False

    def test_calculate_frequency_deviation_alert(self):
        """Test frequency deviation calculation in alert condition."""
        calculator = FrequencyDeviationCalculator(alert_threshold_hz=0.05)

        # Mock grid engine with high frequency
        mock_engine = Mock()
        mock_engine.get_state.return_value.frequency_hz = 60.08  # 0.08 Hz deviation

        result = calculator.calculate_frequency_metrics(mock_engine)

        assert result["current_frequency_hz"] == 60.08
        assert result["frequency_deviation_hz"] == 0.08
        assert result["frequency_status"] == "alert"
        assert result["is_emergency"] is False

    def test_calculate_frequency_deviation_emergency(self):
        """Test frequency deviation calculation in emergency condition."""
        calculator = FrequencyDeviationCalculator(emergency_threshold_hz=0.3)

        # Mock grid engine with very low frequency
        mock_engine = Mock()
        mock_engine.get_state.return_value.frequency_hz = 59.6  # 0.4 Hz deviation

        result = calculator.calculate_frequency_metrics(mock_engine)

        assert result["current_frequency_hz"] == 59.6
        assert result["frequency_deviation_hz"] == 0.4
        assert result["frequency_status"] == "emergency"
        assert result["is_emergency"] is True

    def test_frequency_deviation_statistics(self):
        """Test frequency deviation statistics calculation."""
        calculator = FrequencyDeviationCalculator()

        # Simulate time series of frequency values
        frequency_values = [60.01, 59.98, 60.03, 59.97, 60.02, 60.04, 59.99]

        result = calculator.calculate_frequency_statistics(frequency_values)

        assert abs(result["mean_frequency_hz"] - 60.006) < 0.001
        assert abs(result["std_frequency_hz"] - 0.025) < 0.001
        assert result["min_frequency_hz"] == 59.97
        assert result["max_frequency_hz"] == 60.04
        assert abs(result["mean_absolute_deviation_hz"] - 0.02) < 0.01

    def test_frequency_nadir_and_zenith_tracking(self):
        """Test tracking of frequency nadir and zenith events."""
        calculator = FrequencyDeviationCalculator()

        # Simulate frequency event with nadir and recovery
        frequency_sequence = [60.0, 59.8, 59.7, 59.9, 60.1, 60.0]  # Nadir at 59.7 Hz
        timestamps = [datetime(2024, 1, 1, 12, i, 0) for i in range(6)]

        events = []
        for freq, timestamp in zip(frequency_sequence, timestamps, strict=False):
            mock_engine = Mock()
            mock_engine.get_state.return_value.frequency_hz = freq
            mock_engine.current_time = timestamp

            metrics = calculator.calculate_frequency_metrics(mock_engine)
            events.append(metrics)

        # Analyze events for nadir
        min_freq_event = min(events, key=lambda x: x["current_frequency_hz"])
        max_freq_event = max(events, key=lambda x: x["current_frequency_hz"])

        assert min_freq_event["current_frequency_hz"] == 59.7  # Nadir
        assert max_freq_event["current_frequency_hz"] == 60.1  # Zenith

    def test_frequency_rate_of_change_calculation(self):
        """Test rate of change of frequency calculation."""
        calculator = FrequencyDeviationCalculator()

        # Simulate rapid frequency change
        previous_frequency = 60.0
        current_frequency = 59.5
        time_delta_seconds = 30.0

        rate_of_change = calculator.calculate_frequency_rate_of_change(
            previous_frequency, current_frequency, time_delta_seconds
        )

        # Rate of change: (59.5 - 60.0) / 30 = -0.0167 Hz/s
        assert abs(rate_of_change - (-0.0167)) < 0.001


class TestVoltageDeviationMetrics:
    """Test voltage deviation calculation and tracking."""

    def test_voltage_deviation_calculator_initialization(self):
        """Test VoltageDeviationCalculator initialization."""
        calculator = VoltageDeviationCalculator(
            nominal_voltage_kv=138.0,
            voltage_deadband_percent=2.0,
            alert_threshold_percent=5.0,
            emergency_threshold_percent=10.0,
        )

        assert calculator.nominal_voltage_kv == 138.0
        assert calculator.voltage_deadband_percent == 2.0
        assert calculator.alert_threshold_percent == 5.0
        assert calculator.emergency_threshold_percent == 10.0

    def test_calculate_voltage_deviation_normal(self):
        """Test voltage deviation calculation under normal conditions."""
        calculator = VoltageDeviationCalculator(nominal_voltage_kv=138.0)

        # Mock grid engine with normal voltage
        mock_engine = Mock()
        mock_node = Mock()
        mock_node.node_id = "test_node"
        mock_node.current_voltage_kv = 139.0  # Within normal range
        mock_node.voltage_kv = 138.0  # Nominal
        mock_engine.nodes = {"test_node": mock_node}

        result = calculator.calculate_voltage_metrics(mock_engine)

        node_metrics = result["node_metrics"]["test_node"]
        assert node_metrics["current_voltage_kv"] == 139.0
        assert node_metrics["nominal_voltage_kv"] == 138.0
        assert abs(node_metrics["voltage_deviation_kv"] - 1.0) < 0.001
        assert abs(node_metrics["voltage_deviation_percent"] - 0.725) < 0.001  # 1/138 * 100
        assert node_metrics["voltage_status"] == "normal"

    def test_calculate_voltage_deviation_alert(self):
        """Test voltage deviation calculation in alert condition."""
        calculator = VoltageDeviationCalculator(nominal_voltage_kv=138.0, alert_threshold_percent=3.0)

        # Mock grid engine with high voltage
        mock_engine = Mock()
        mock_node = Mock()
        mock_node.node_id = "test_node"
        mock_node.current_voltage_kv = 145.0  # High voltage
        mock_node.voltage_kv = 138.0
        mock_engine.nodes = {"test_node": mock_node}

        result = calculator.calculate_voltage_metrics(mock_engine)

        node_metrics = result["node_metrics"]["test_node"]
        assert node_metrics["current_voltage_kv"] == 145.0
        assert abs(node_metrics["voltage_deviation_percent"] - 5.07) < 0.01  # 7/138 * 100
        assert node_metrics["voltage_status"] == "alert"

    def test_calculate_voltage_deviation_emergency(self):
        """Test voltage deviation calculation in emergency condition."""
        calculator = VoltageDeviationCalculator(nominal_voltage_kv=138.0, emergency_threshold_percent=8.0)

        # Mock grid engine with very low voltage
        mock_engine = Mock()
        mock_node = Mock()
        mock_node.node_id = "test_node"
        mock_node.current_voltage_kv = 125.0  # Very low voltage
        mock_node.voltage_kv = 138.0
        mock_engine.nodes = {"test_node": mock_node}

        result = calculator.calculate_voltage_metrics(mock_engine)

        node_metrics = result["node_metrics"]["test_node"]
        assert node_metrics["current_voltage_kv"] == 125.0
        assert abs(node_metrics["voltage_deviation_percent"] - (-9.42)) < 0.01  # -13/138 * 100
        assert node_metrics["voltage_status"] == "emergency"
        assert result["system_summary"]["emergency_nodes"] == 1

    def test_voltage_metrics_multiple_nodes(self):
        """Test voltage metrics calculation for multiple nodes."""
        calculator = VoltageDeviationCalculator(nominal_voltage_kv=138.0)

        # Mock grid engine with multiple nodes
        mock_engine = Mock()

        node1 = Mock()
        node1.node_id = "node_1"
        node1.current_voltage_kv = 139.0
        node1.voltage_kv = 138.0

        node2 = Mock()
        node2.node_id = "node_2"
        node2.current_voltage_kv = 142.0
        node2.voltage_kv = 138.0

        node3 = Mock()
        node3.node_id = "node_3"
        node3.current_voltage_kv = 135.0
        node3.voltage_kv = 138.0

        mock_engine.nodes = {"node_1": node1, "node_2": node2, "node_3": node3}

        result = calculator.calculate_voltage_metrics(mock_engine)

        # Verify individual node metrics
        assert len(result["node_metrics"]) == 3
        assert "node_1" in result["node_metrics"]
        assert "node_2" in result["node_metrics"]
        assert "node_3" in result["node_metrics"]

        # Verify system summary
        summary = result["system_summary"]
        assert summary["total_nodes"] == 3
        assert abs(summary["mean_voltage_deviation_percent"] - 0.483) < 0.01
        assert summary["max_voltage_deviation_percent"] == 2.899  # node_2
        assert summary["min_voltage_deviation_percent"] == -2.174  # node_3

    def test_voltage_regulation_effectiveness(self):
        """Test calculation of voltage regulation effectiveness."""
        calculator = VoltageDeviationCalculator()

        # Simulate before and after voltage regulation
        before_voltages = [135.0, 142.0, 133.0, 144.0]  # High deviation
        after_voltages = [137.5, 138.5, 137.0, 139.0]  # Low deviation
        nominal_voltage = 138.0

        effectiveness = calculator.calculate_regulation_effectiveness(before_voltages, after_voltages, nominal_voltage)

        assert effectiveness["improvement_percentage"] > 0
        assert effectiveness["before_mean_deviation"] > effectiveness["after_mean_deviation"]
        assert effectiveness["regulation_success"] is True


class TestMetricsCollectorIntegration:
    """Test integration of curtailment and Hz/V deviation metrics with MetricsCollector."""

    def test_curtailment_metrics_hook_integration(self):
        """Test curtailment metrics hook integration with MetricsCollector."""
        import tempfile
        from pathlib import Path

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create metrics collector
            collector = MetricsCollector(log_directory=Path(tmp_dir), enable_time_series_logging=True)

            # Create mock engine with renewable assets
            mock_engine = Mock()
            mock_engine.current_time = datetime(2024, 1, 1, 12, 0, 0)

            # Create mock solar asset with curtailment
            solar_asset = Mock()
            solar_asset.asset_type = AssetType.SOLAR
            solar_asset.current_output_mw = 80.0
            solar_asset.get_theoretical_max_power.return_value = 100.0
            solar_asset.is_online.return_value = True
            solar_asset.curtailment_factor = 0.8

            mock_engine.get_all_assets.return_value = [solar_asset]

            # Register default hooks
            collector.register_default_hooks()

            # Collect metrics
            collector.collect_metrics(mock_engine)

            # Verify curtailment metrics were collected
            data = collector.time_series_data[0]
            assert "curtailment_metrics" in data

            curtailment_metrics = data["curtailment_metrics"]
            assert "curtailment_percentage" in curtailment_metrics
            assert "curtailed_energy_mw" in curtailment_metrics
            assert curtailment_metrics["curtailment_percentage"] == 20.0  # 20 MW / 100 MW * 100

    def test_frequency_deviation_metrics_hook_integration(self):
        """Test frequency deviation metrics hook integration with MetricsCollector."""
        import tempfile
        from pathlib import Path

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create metrics collector
            collector = MetricsCollector(log_directory=Path(tmp_dir), enable_time_series_logging=True)

            # Create mock engine with frequency deviation
            mock_engine = Mock()
            mock_engine.current_time = datetime(2024, 1, 1, 12, 0, 0)
            mock_engine.get_state.return_value.frequency_hz = 59.95  # Low frequency
            mock_engine.get_all_assets.return_value = []

            # Register default hooks
            collector.register_default_hooks()

            # Collect metrics
            collector.collect_metrics(mock_engine)

            # Verify frequency metrics were collected
            data = collector.time_series_data[0]
            assert "frequency_deviation_metrics" in data

            frequency_metrics = data["frequency_deviation_metrics"]
            assert "current_frequency_hz" in frequency_metrics
            assert "frequency_deviation_hz" in frequency_metrics
            assert frequency_metrics["current_frequency_hz"] == 59.95
            assert frequency_metrics["frequency_deviation_hz"] == 0.05

    def test_voltage_deviation_metrics_hook_integration(self):
        """Test voltage deviation metrics hook integration with MetricsCollector."""
        import tempfile
        from pathlib import Path

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create metrics collector
            collector = MetricsCollector(log_directory=Path(tmp_dir), enable_time_series_logging=True)

            # Create mock engine with voltage deviation
            mock_engine = Mock()
            mock_engine.current_time = datetime(2024, 1, 1, 12, 0, 0)

            # Create mock node with voltage deviation
            mock_node = Mock()
            mock_node.node_id = "test_node"
            mock_node.current_voltage_kv = 142.0  # High voltage
            mock_node.voltage_kv = 138.0

            mock_engine.nodes = {"test_node": mock_node}
            mock_engine.get_all_assets.return_value = []

            # Register default hooks
            collector.register_default_hooks()

            # Collect metrics
            collector.collect_metrics(mock_engine)

            # Verify voltage metrics were collected
            data = collector.time_series_data[0]
            assert "voltage_deviation_metrics" in data

            voltage_metrics = data["voltage_deviation_metrics"]
            assert "node_metrics" in voltage_metrics
            assert "system_summary" in voltage_metrics
            assert "test_node" in voltage_metrics["node_metrics"]

            node_metrics = voltage_metrics["node_metrics"]["test_node"]
            assert node_metrics["current_voltage_kv"] == 142.0
            assert abs(node_metrics["voltage_deviation_percent"] - 2.899) < 0.01
