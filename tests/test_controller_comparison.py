"""Tests for controller comparison framework."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from psireg.sim.engine import GridEngine


class TestControllerComparison:
    """Test controller comparison framework."""

    def test_comparison_framework_creation(self):
        """Test that comparison framework can be created."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()
        assert comparison is not None

    def test_comparison_framework_controller_registration(self):
        """Test controller registration in comparison framework."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()

        # Mock controllers
        rule_controller = Mock()
        rule_controller.name = "Rule-Based"

        ml_controller = Mock()
        ml_controller.name = "ML-Only"

        swarm_controller = Mock()
        swarm_controller.name = "Swarm-Only"

        # Register controllers
        comparison.register_controller("rule", rule_controller)
        comparison.register_controller("ml", ml_controller)
        comparison.register_controller("swarm", swarm_controller)

        assert len(comparison.controllers) == 3
        assert "rule" in comparison.controllers
        assert "ml" in comparison.controllers
        assert "swarm" in comparison.controllers

    def test_comparison_framework_scenario_setup(self):
        """Test scenario setup for comparison."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()
        grid_engine = Mock(spec=GridEngine)

        # Test scenario configuration
        scenario_config = {
            "name": "peak_demand",
            "duration_hours": 24,
            "weather_conditions": ["CLEAR", "CLOUDY"],
            "load_profile": "residential_peak",
            "renewable_variability": "moderate",
        }

        result = comparison.setup_scenario(scenario_config, grid_engine)
        assert result is True

    def test_comparison_framework_execution(self):
        """Test comparison framework execution."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()

        # Mock controllers
        controllers = {}
        for name in ["rule", "ml", "swarm"]:
            controller = Mock()
            controller.initialize.return_value = True
            controller.get_performance_metrics.return_value = {
                "efficiency": 0.85 + name.__hash__() % 10 * 0.01,
                "frequency_deviation_hz": 0.02,
                "response_time_s": 1.5,
            }
            controllers[name] = controller
            comparison.register_controller(name, controller)

        grid_engine = Mock(spec=GridEngine)
        scenario_config = {"name": "test_scenario", "duration_hours": 1}

        results = comparison.run_comparison(scenario_config, grid_engine)

        assert isinstance(results, dict)
        assert len(results) == 3
        assert "rule" in results
        assert "ml" in results
        assert "swarm" in results

    def test_comparison_framework_metrics_collection(self):
        """Test metrics collection during comparison."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()

        # Mock controller with detailed metrics
        controller = Mock()
        controller.initialize.return_value = True
        controller.get_performance_metrics.return_value = {
            "efficiency": 0.88,
            "frequency_deviation_hz": 0.015,
            "power_balance_mw": 2.3,
            "response_time_s": 1.2,
            "energy_losses_mwh": 0.5,
            "renewable_utilization": 0.92,
            "cost_per_mwh": 45.0,
        }

        comparison.register_controller("test", controller)

        grid_engine = Mock(spec=GridEngine)
        scenario_config = {"name": "test", "duration_hours": 1}

        results = comparison.run_comparison(scenario_config, grid_engine)

        # Verify comprehensive metrics collection
        test_results = results["test"]
        assert "efficiency" in test_results
        assert "frequency_deviation_hz" in test_results
        assert "response_time_s" in test_results
        assert "cost_per_mwh" in test_results

    def test_comparison_framework_performance_analysis(self):
        """Test performance analysis and ranking."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()

        # Mock results data
        results_data = {
            "rule": {"efficiency": 0.82, "frequency_deviation_hz": 0.025, "response_time_s": 2.0, "cost_per_mwh": 50.0},
            "ml": {"efficiency": 0.89, "frequency_deviation_hz": 0.018, "response_time_s": 1.5, "cost_per_mwh": 48.0},
            "swarm": {
                "efficiency": 0.91,
                "frequency_deviation_hz": 0.020,
                "response_time_s": 1.8,
                "cost_per_mwh": 46.0,
            },
        }

        analysis = comparison.analyze_performance(results_data)

        assert isinstance(analysis, dict)
        assert "rankings" in analysis
        assert "summary_statistics" in analysis
        assert "best_performers" in analysis

    def test_comparison_framework_statistical_analysis(self):
        """Test statistical analysis of comparison results."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()

        # Mock multiple run results
        multiple_results = []
        for _run in range(10):
            run_results = {
                "rule": {"efficiency": 0.82 + np.random.normal(0, 0.02)},
                "ml": {"efficiency": 0.89 + np.random.normal(0, 0.02)},
                "swarm": {"efficiency": 0.91 + np.random.normal(0, 0.02)},
            }
            multiple_results.append(run_results)

        stats = comparison.statistical_analysis(multiple_results)

        assert isinstance(stats, dict)
        assert "mean_performance" in stats
        assert "confidence_intervals" in stats
        assert "significance_tests" in stats

    def test_comparison_framework_scenario_variations(self):
        """Test comparison across multiple scenario variations."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()

        # Mock controller
        controller = Mock()
        controller.initialize.return_value = True
        controller.get_performance_metrics.return_value = {"efficiency": 0.85}
        comparison.register_controller("test", controller)

        # Multiple scenarios
        scenarios = [
            {"name": "normal_operation", "duration_hours": 24},
            {"name": "peak_demand", "duration_hours": 24},
            {"name": "renewable_surge", "duration_hours": 24},
            {"name": "grid_outage", "duration_hours": 12},
        ]

        grid_engine = Mock(spec=GridEngine)

        results = comparison.run_multi_scenario_comparison(scenarios, grid_engine)

        assert isinstance(results, dict)
        assert len(results) == 4
        for scenario in scenarios:
            assert scenario["name"] in results

    def test_comparison_framework_report_generation(self):
        """Test comparison report generation."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()

        # Mock results data
        results_data = {
            "rule": {"efficiency": 0.82, "cost_per_mwh": 50.0},
            "ml": {"efficiency": 0.89, "cost_per_mwh": 48.0},
            "swarm": {"efficiency": 0.91, "cost_per_mwh": 46.0},
        }

        report = comparison.generate_report(results_data)

        assert isinstance(report, dict)
        assert "executive_summary" in report
        assert "detailed_analysis" in report
        assert "recommendations" in report

    def test_comparison_framework_visualization_data(self):
        """Test visualization data preparation."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()

        # Mock time series data
        time_series_data = {
            "rule": {
                "timestamps": pd.date_range("2023-01-01", periods=24, freq="H"),
                "frequency_hz": np.random.normal(60.0, 0.02, 24),
                "power_balance_mw": np.random.normal(0, 5, 24),
            },
            "ml": {
                "timestamps": pd.date_range("2023-01-01", periods=24, freq="H"),
                "frequency_hz": np.random.normal(60.0, 0.015, 24),
                "power_balance_mw": np.random.normal(0, 3, 24),
            },
            "swarm": {
                "timestamps": pd.date_range("2023-01-01", periods=24, freq="H"),
                "frequency_hz": np.random.normal(60.0, 0.018, 24),
                "power_balance_mw": np.random.normal(0, 4, 24),
            },
        }

        viz_data = comparison.prepare_visualization_data(time_series_data)

        assert isinstance(viz_data, dict)
        assert "performance_comparison" in viz_data
        assert "time_series_plots" in viz_data

    def test_comparison_framework_parallel_execution(self):
        """Test parallel execution of controller comparisons."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison(parallel=True)

        # Mock controllers
        for name in ["rule", "ml", "swarm"]:
            controller = Mock()
            controller.initialize.return_value = True
            controller.get_performance_metrics.return_value = {"efficiency": 0.85}
            comparison.register_controller(name, controller)

        grid_engine = Mock(spec=GridEngine)
        scenario_config = {"name": "test", "duration_hours": 1}

        with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.map.return_value = [
                {"efficiency": 0.82},
                {"efficiency": 0.89},
                {"efficiency": 0.91},
            ]

            results = comparison.run_comparison(scenario_config, grid_engine)

            assert isinstance(results, dict)


class TestComparisonMetrics:
    """Test comparison metrics and evaluation."""

    def test_efficiency_metric_calculation(self):
        """Test efficiency metric calculation."""
        from psireg.controllers.comparison import ComparisonMetrics

        metrics = ComparisonMetrics()

        # Mock simulation data
        grid_data = {
            "total_generation_mwh": 1000,
            "total_load_mwh": 950,
            "energy_losses_mwh": 30,
            "renewable_generation_mwh": 600,
        }

        efficiency = metrics.calculate_efficiency(grid_data)

        assert 0.0 <= efficiency <= 1.0
        assert isinstance(efficiency, float)

    def test_frequency_stability_metric(self):
        """Test frequency stability metric calculation."""
        from psireg.controllers.comparison import ComparisonMetrics

        metrics = ComparisonMetrics()

        # Mock frequency data
        frequency_data = np.array([59.98, 60.01, 59.99, 60.02, 59.97, 60.03, 59.98, 60.00])

        stability = metrics.calculate_frequency_stability(frequency_data)

        assert isinstance(stability, float)
        assert stability >= 0.0

    def test_economic_performance_metric(self):
        """Test economic performance metric calculation."""
        from psireg.controllers.comparison import ComparisonMetrics

        metrics = ComparisonMetrics()

        # Mock economic data
        economic_data = {
            "total_generation_cost": 45000,  # $
            "total_energy_mwh": 1000,
            "demand_response_savings": 2000,
            "renewable_incentives": 5000,
        }

        cost_per_mwh = metrics.calculate_economic_performance(economic_data)

        assert isinstance(cost_per_mwh, float)
        assert cost_per_mwh > 0

    def test_renewable_integration_metric(self):
        """Test renewable integration metric calculation."""
        from psireg.controllers.comparison import ComparisonMetrics

        metrics = ComparisonMetrics()

        # Mock renewable data
        renewable_data = {
            "renewable_generation_mwh": 650,
            "renewable_capacity_mwh": 800,
            "renewable_curtailment_mwh": 50,
            "total_generation_mwh": 1000,
        }

        integration_score = metrics.calculate_renewable_integration(renewable_data)

        assert 0.0 <= integration_score <= 1.0
        assert isinstance(integration_score, float)

    def test_response_time_metric(self):
        """Test response time metric calculation."""
        from psireg.controllers.comparison import ComparisonMetrics

        metrics = ComparisonMetrics()

        # Mock response time data
        response_times = [1.2, 0.8, 1.5, 0.9, 1.1, 2.0, 1.3, 0.7]

        avg_response_time = metrics.calculate_response_time(response_times)

        assert isinstance(avg_response_time, float)
        assert avg_response_time > 0

    def test_composite_score_calculation(self):
        """Test composite performance score calculation."""
        from psireg.controllers.comparison import ComparisonMetrics

        metrics = ComparisonMetrics()

        # Mock individual metrics
        individual_metrics = {
            "efficiency": 0.88,
            "frequency_stability": 0.92,
            "economic_performance": 0.85,
            "renewable_integration": 0.90,
            "response_time_score": 0.87,
        }

        # Custom weights
        weights = {
            "efficiency": 0.25,
            "frequency_stability": 0.25,
            "economic_performance": 0.20,
            "renewable_integration": 0.20,
            "response_time_score": 0.10,
        }

        composite_score = metrics.calculate_composite_score(individual_metrics, weights)

        assert 0.0 <= composite_score <= 1.0
        assert isinstance(composite_score, float)


class TestComparisonScenarios:
    """Test specific comparison scenarios."""

    def test_storm_day_scenario_comparison(self):
        """Test controller comparison during storm day scenario."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()

        # Mock controllers with storm-specific responses
        rule_controller = Mock()
        rule_controller.initialize.return_value = True
        rule_controller.get_performance_metrics.return_value = {
            "efficiency": 0.78,  # Lower efficiency during storm
            "frequency_deviation_hz": 0.035,
            "emergency_response_time_s": 3.0,
        }

        ml_controller = Mock()
        ml_controller.initialize.return_value = True
        ml_controller.get_performance_metrics.return_value = {
            "efficiency": 0.84,  # Better adaptation
            "frequency_deviation_hz": 0.022,
            "emergency_response_time_s": 2.2,
        }

        swarm_controller = Mock()
        swarm_controller.initialize.return_value = True
        swarm_controller.get_performance_metrics.return_value = {
            "efficiency": 0.87,  # Best coordination
            "frequency_deviation_hz": 0.025,
            "emergency_response_time_s": 1.8,
        }

        comparison.register_controller("rule", rule_controller)
        comparison.register_controller("ml", ml_controller)
        comparison.register_controller("swarm", swarm_controller)

        grid_engine = Mock(spec=GridEngine)
        storm_scenario = {
            "name": "storm_day",
            "duration_hours": 16,
            "weather_conditions": ["STORMY"],
            "emergency_events": ["wind_turbine_shutdown", "transmission_line_trip"],
        }

        results = comparison.run_comparison(storm_scenario, grid_engine)

        # Verify storm-specific metrics are captured
        assert "emergency_response_time_s" in results["rule"]
        assert "emergency_response_time_s" in results["ml"]
        assert "emergency_response_time_s" in results["swarm"]

    def test_renewable_surge_scenario_comparison(self):
        """Test controller comparison during renewable surge scenario."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()

        # Mock controllers with different renewable handling
        controllers_data = {
            "rule": {"renewable_utilization": 0.85, "curtailment_mwh": 150},
            "ml": {"renewable_utilization": 0.92, "curtailment_mwh": 80},
            "swarm": {"renewable_utilization": 0.94, "curtailment_mwh": 60},
        }

        for name, data in controllers_data.items():
            controller = Mock()
            controller.initialize.return_value = True
            controller.get_performance_metrics.return_value = data
            comparison.register_controller(name, controller)

        grid_engine = Mock(spec=GridEngine)
        renewable_scenario = {
            "name": "renewable_surge",
            "duration_hours": 24,
            "renewable_capacity_factor": 0.9,  # High renewable output
            "storage_availability": 0.7,
        }

        results = comparison.run_comparison(renewable_scenario, grid_engine)

        # Verify renewable-specific metrics
        for controller_name in controllers_data.keys():
            assert "renewable_utilization" in results[controller_name]

    def test_peak_demand_scenario_comparison(self):
        """Test controller comparison during peak demand scenario."""
        # This will test peak demand handling once implemented
        pass

    def test_grid_outage_scenario_comparison(self):
        """Test controller comparison during grid outage scenario."""
        # This will test outage recovery once implemented
        pass


class TestComparisonIntegration:
    """Test comparison framework integration."""

    def test_comparison_with_real_controllers(self):
        """Test comparison framework with actual controller implementations."""
        # This will test real controller integration once implemented
        pass

    def test_comparison_data_export(self):
        """Test comparison data export functionality."""
        from psireg.controllers.comparison import ControllerComparison

        comparison = ControllerComparison()

        # Mock results
        results_data = {"rule": {"efficiency": 0.82}, "ml": {"efficiency": 0.89}, "swarm": {"efficiency": 0.91}}

        # Test CSV export
        csv_data = comparison.export_to_csv(results_data)
        assert isinstance(csv_data, str)

        # Test JSON export
        json_data = comparison.export_to_json(results_data)
        assert isinstance(json_data, str)

    def test_comparison_reproducibility(self):
        """Test comparison framework reproducibility."""
        # This will test result reproducibility once implemented
        pass
