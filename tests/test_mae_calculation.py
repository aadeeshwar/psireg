"""Tests for MAE (Mean Absolute Error) calculation framework for prediction accuracy."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
from psireg.sim.metrics import MAECalculator, MetricsCollector, PredictionTracker


class TestMAECalculator:
    """Test MAE (Mean Absolute Error) calculation framework."""

    def test_mae_calculator_initialization(self):
        """Test MAECalculator initialization."""
        calculator = MAECalculator(
            prediction_horizon_hours=24,
            min_samples_for_mae=5,
            track_prediction_types=["power_generation", "frequency", "load_demand"],
        )

        assert calculator.prediction_horizon_hours == 24
        assert calculator.min_samples_for_mae == 5
        assert calculator.track_prediction_types == ["power_generation", "frequency", "load_demand"]
        assert len(calculator.prediction_history) == 0
        assert len(calculator.actual_history) == 0

    def test_add_prediction_data(self):
        """Test adding prediction data to MAE calculator."""
        calculator = MAECalculator()

        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        predictions = {"power_generation": 150.0, "frequency": 60.02, "load_demand": 140.0}

        calculator.add_prediction(timestamp, predictions, prediction_horizon_minutes=60)

        assert len(calculator.prediction_history) == 1

        prediction_entry = calculator.prediction_history[0]
        assert prediction_entry["timestamp"] == timestamp
        assert prediction_entry["predictions"] == predictions
        assert prediction_entry["prediction_horizon_minutes"] == 60
        assert prediction_entry["target_timestamp"] == timestamp + timedelta(minutes=60)

    def test_add_actual_data(self):
        """Test adding actual data to MAE calculator."""
        calculator = MAECalculator()

        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        actuals = {"power_generation": 145.0, "frequency": 60.01, "load_demand": 142.0}

        calculator.add_actual(timestamp, actuals)

        assert len(calculator.actual_history) == 1

        actual_entry = calculator.actual_history[0]
        assert actual_entry["timestamp"] == timestamp
        assert actual_entry["actuals"] == actuals

    def test_calculate_mae_simple(self):
        """Test simple MAE calculation with matched prediction-actual pairs."""
        calculator = MAECalculator(min_samples_for_mae=3)

        # Add prediction-actual pairs
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Prediction at 12:00 for 13:00
        calculator.add_prediction(
            base_time, {"power_generation": 150.0, "frequency": 60.05}, prediction_horizon_minutes=60
        )

        # Prediction at 12:15 for 13:15
        calculator.add_prediction(
            base_time + timedelta(minutes=15),
            {"power_generation": 155.0, "frequency": 60.03},
            prediction_horizon_minutes=60,
        )

        # Prediction at 12:30 for 13:30
        calculator.add_prediction(
            base_time + timedelta(minutes=30),
            {"power_generation": 160.0, "frequency": 60.02},
            prediction_horizon_minutes=60,
        )

        # Add corresponding actual values
        calculator.add_actual(base_time + timedelta(hours=1), {"power_generation": 148.0, "frequency": 60.02})  # 13:00

        calculator.add_actual(
            base_time + timedelta(hours=1, minutes=15), {"power_generation": 152.0, "frequency": 60.01}  # 13:15
        )

        calculator.add_actual(
            base_time + timedelta(hours=1, minutes=30), {"power_generation": 158.0, "frequency": 60.03}  # 13:30
        )

        # Calculate MAE
        mae_results = calculator.calculate_mae()

        # Expected MAE for power_generation: (|150-148| + |155-152| + |160-158|) / 3 = (2+3+2)/3 = 2.33
        assert abs(mae_results["power_generation"]["mae"] - 2.33) < 0.01

        # Expected MAE for frequency: (|60.05-60.02| + |60.03-60.01| + |60.02-60.03|) / 3 = (0.03+0.02+0.01)/3 = 0.02
        assert abs(mae_results["frequency"]["mae"] - 0.02) < 0.001

        assert mae_results["power_generation"]["sample_count"] == 3
        assert mae_results["frequency"]["sample_count"] == 3

    def test_calculate_mae_with_missing_data(self):
        """Test MAE calculation when some actual data is missing."""
        calculator = MAECalculator(min_samples_for_mae=2)

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Add predictions
        calculator.add_prediction(base_time, {"power_generation": 150.0}, prediction_horizon_minutes=60)

        calculator.add_prediction(
            base_time + timedelta(minutes=15), {"power_generation": 155.0}, prediction_horizon_minutes=60
        )

        calculator.add_prediction(
            base_time + timedelta(minutes=30), {"power_generation": 160.0}, prediction_horizon_minutes=60
        )

        # Add only some actual values (simulating missing data)
        calculator.add_actual(base_time + timedelta(hours=1), {"power_generation": 148.0})

        calculator.add_actual(base_time + timedelta(hours=1, minutes=30), {"power_generation": 158.0})

        # Calculate MAE
        mae_results = calculator.calculate_mae()

        # Should calculate MAE for available pairs only
        assert mae_results["power_generation"]["sample_count"] == 2
        assert abs(mae_results["power_generation"]["mae"] - 2.0) < 0.01  # (|150-148| + |160-158|) / 2 = 2.0

    def test_calculate_mae_insufficient_samples(self):
        """Test MAE calculation with insufficient samples."""
        calculator = MAECalculator(min_samples_for_mae=5)

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Add only 2 prediction-actual pairs (less than min_samples_for_mae)
        calculator.add_prediction(base_time, {"power_generation": 150.0}, prediction_horizon_minutes=60)
        calculator.add_prediction(
            base_time + timedelta(minutes=15), {"power_generation": 155.0}, prediction_horizon_minutes=60
        )

        calculator.add_actual(base_time + timedelta(hours=1), {"power_generation": 148.0})
        calculator.add_actual(base_time + timedelta(hours=1, minutes=15), {"power_generation": 152.0})

        # Calculate MAE
        mae_results = calculator.calculate_mae()

        # Should indicate insufficient samples
        assert mae_results["power_generation"]["mae"] is None
        assert mae_results["power_generation"]["sample_count"] == 2
        assert mae_results["power_generation"]["insufficient_samples"] is True

    def test_calculate_mae_by_horizon(self):
        """Test MAE calculation broken down by prediction horizon."""
        calculator = MAECalculator()

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Add predictions with different horizons
        # 15-minute horizon predictions
        calculator.add_prediction(base_time, {"power_generation": 150.0}, prediction_horizon_minutes=15)
        calculator.add_prediction(
            base_time + timedelta(minutes=15), {"power_generation": 155.0}, prediction_horizon_minutes=15
        )

        # 60-minute horizon predictions
        calculator.add_prediction(base_time, {"power_generation": 160.0}, prediction_horizon_minutes=60)
        calculator.add_prediction(
            base_time + timedelta(minutes=15), {"power_generation": 165.0}, prediction_horizon_minutes=60
        )

        # Add actual values
        calculator.add_actual(base_time + timedelta(minutes=15), {"power_generation": 152.0})
        calculator.add_actual(base_time + timedelta(minutes=30), {"power_generation": 153.0})
        calculator.add_actual(base_time + timedelta(hours=1), {"power_generation": 158.0})
        calculator.add_actual(base_time + timedelta(hours=1, minutes=15), {"power_generation": 162.0})

        # Calculate MAE by horizon
        mae_by_horizon = calculator.calculate_mae_by_horizon()

        assert 15 in mae_by_horizon
        assert 60 in mae_by_horizon

        # MAE for 15-minute horizon should be different from 60-minute horizon
        mae_15min = mae_by_horizon[15]["power_generation"]["mae"]
        mae_60min = mae_by_horizon[60]["power_generation"]["mae"]

        assert mae_15min is not None
        assert mae_60min is not None
        assert mae_15min != mae_60min

    def test_calculate_mape(self):
        """Test Mean Absolute Percentage Error (MAPE) calculation."""
        calculator = MAECalculator()

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Add prediction-actual pairs
        calculator.add_prediction(base_time, {"power_generation": 100.0}, prediction_horizon_minutes=60)
        calculator.add_prediction(
            base_time + timedelta(minutes=15), {"power_generation": 200.0}, prediction_horizon_minutes=60
        )

        calculator.add_actual(base_time + timedelta(hours=1), {"power_generation": 110.0})
        calculator.add_actual(base_time + timedelta(hours=1, minutes=15), {"power_generation": 180.0})

        # Calculate MAPE
        mape_results = calculator.calculate_mape()

        # Expected MAPE: (|100-110|/110 + |200-180|/180) / 2 = (0.091 + 0.111) / 2 = 0.101 = 10.1%
        assert abs(mape_results["power_generation"]["mape"] - 10.1) < 0.1

    def test_calculate_rmse(self):
        """Test Root Mean Square Error (RMSE) calculation."""
        calculator = MAECalculator()

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Add prediction-actual pairs
        predictions = [100.0, 150.0, 200.0]
        actuals = [110.0, 140.0, 195.0]

        for i, (pred, actual) in enumerate(zip(predictions, actuals, strict=False)):
            calculator.add_prediction(
                base_time + timedelta(minutes=i * 15), {"power_generation": pred}, prediction_horizon_minutes=60
            )
            calculator.add_actual(base_time + timedelta(hours=1, minutes=i * 15), {"power_generation": actual})

        # Calculate RMSE
        rmse_results = calculator.calculate_rmse()

        # Expected RMSE: sqrt(((100-110)^2 + (150-140)^2 + (200-195)^2) / 3) = sqrt((100+100+25)/3) = sqrt(75) = 8.66
        assert abs(rmse_results["power_generation"]["rmse"] - 8.66) < 0.01

    def test_prediction_accuracy_trends(self):
        """Test calculation of prediction accuracy trends over time."""
        calculator = MAECalculator()

        # Simulate deteriorating prediction accuracy over time
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        for hour in range(24):  # 24 hours of data
            timestamp = base_time + timedelta(hours=hour)

            # Predictions get worse over time (more error)
            error_magnitude = hour * 0.5  # Increasing error
            predicted_value = 100.0
            actual_value = 100.0 + error_magnitude

            calculator.add_prediction(timestamp, {"power_generation": predicted_value}, prediction_horizon_minutes=60)

            calculator.add_actual(timestamp + timedelta(hours=1), {"power_generation": actual_value})

        # Calculate accuracy trends
        trends = calculator.calculate_accuracy_trends(window_hours=6)

        # Should show deteriorating accuracy
        assert len(trends) > 1
        first_window_mae = trends[0]["power_generation"]["mae"]
        last_window_mae = trends[-1]["power_generation"]["mae"]

        assert last_window_mae > first_window_mae  # Accuracy getting worse

    def test_outlier_detection(self):
        """Test outlier detection in prediction errors."""
        calculator = MAECalculator()

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Add mostly accurate predictions with one outlier
        predictions = [100.0, 102.0, 98.0, 101.0, 150.0, 99.0]  # 150.0 is outlier
        actuals = [101.0, 103.0, 97.0, 102.0, 105.0, 100.0]

        for i, (pred, actual) in enumerate(zip(predictions, actuals, strict=False)):
            calculator.add_prediction(
                base_time + timedelta(minutes=i * 15), {"power_generation": pred}, prediction_horizon_minutes=60
            )
            calculator.add_actual(base_time + timedelta(hours=1, minutes=i * 15), {"power_generation": actual})

        # Detect outliers
        outliers = calculator.detect_outliers(threshold_std=2.0)

        assert len(outliers["power_generation"]) > 0
        # The outlier should be the prediction of 150.0 vs actual 105.0 (error = 45.0)
        outlier_errors = [outlier["error"] for outlier in outliers["power_generation"]]
        assert max(outlier_errors) == 45.0


class TestPredictionTracker:
    """Test prediction tracking functionality."""

    def test_prediction_tracker_initialization(self):
        """Test PredictionTracker initialization."""
        tracker = PredictionTracker(prediction_source="test_predictor", track_types=["power", "frequency"])

        assert tracker.prediction_source == "test_predictor"
        assert tracker.track_types == ["power", "frequency"]
        assert len(tracker.predictions) == 0

    def test_track_prediction(self):
        """Test tracking a prediction."""
        tracker = PredictionTracker("test_predictor")

        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        prediction_data = {"power_generation": 150.0, "frequency": 60.02}

        tracker.track_prediction(timestamp, prediction_data, horizon_minutes=60)

        assert len(tracker.predictions) == 1

        prediction = tracker.predictions[0]
        assert prediction["timestamp"] == timestamp
        assert prediction["prediction_data"] == prediction_data
        assert prediction["horizon_minutes"] == 60
        assert prediction["prediction_id"] is not None

    def test_get_predictions_for_timeframe(self):
        """Test retrieving predictions for a specific timeframe."""
        tracker = PredictionTracker("test_predictor")

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Track predictions over 3 hours
        for hour in range(3):
            timestamp = base_time + timedelta(hours=hour)
            tracker.track_prediction(timestamp, {"power_generation": 100.0 + hour * 10}, horizon_minutes=60)

        # Get predictions for middle hour only
        start_time = base_time + timedelta(hours=0.5)
        end_time = base_time + timedelta(hours=1.5)

        predictions_in_range = tracker.get_predictions_for_timeframe(start_time, end_time)

        # Should return only the prediction from hour 1
        assert len(predictions_in_range) == 1
        assert predictions_in_range[0]["prediction_data"]["power_generation"] == 110.0

    def test_compare_with_actuals(self):
        """Test comparison of predictions with actual values."""
        tracker = PredictionTracker("test_predictor")

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Track predictions
        tracker.track_prediction(base_time, {"power_generation": 150.0}, horizon_minutes=60)
        tracker.track_prediction(base_time + timedelta(minutes=15), {"power_generation": 155.0}, horizon_minutes=60)

        # Create actual values
        actuals = [
            {"timestamp": base_time + timedelta(hours=1), "power_generation": 148.0},
            {"timestamp": base_time + timedelta(hours=1, minutes=15), "power_generation": 152.0},
        ]

        # Compare with actuals
        comparison = tracker.compare_with_actuals(actuals)

        assert len(comparison) == 2
        assert abs(comparison[0]["error"]["power_generation"] - 2.0) < 0.01  # |150-148|
        assert abs(comparison[1]["error"]["power_generation"] - 3.0) < 0.01  # |155-152|


class TestMAEIntegrationWithPredictiveLayer:
    """Test MAE calculation integration with PredictiveLayer."""

    def test_mae_integration_with_grid_predictor(self):
        """Test MAE integration with GridPredictor."""
        # Mock GridPredictor
        mock_predictor = Mock()
        mock_predictor.predict.return_value = np.array([150.0, 60.02])
        mock_predictor.get_performance_metrics.return_value = {}

        # Create MAE calculator
        mae_calculator = MAECalculator()

        # Simulate prediction and actual collection
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Make prediction
        prediction_timestamp = base_time
        predicted_values = {"power_generation": 150.0, "frequency": 60.02}
        mae_calculator.add_prediction(prediction_timestamp, predicted_values, prediction_horizon_minutes=60)

        # Add actual values later
        actual_timestamp = base_time + timedelta(hours=1)
        actual_values = {"power_generation": 148.0, "frequency": 60.01}
        mae_calculator.add_actual(actual_timestamp, actual_values)

        # Calculate MAE
        mae_results = mae_calculator.calculate_mae()

        assert mae_results["power_generation"]["mae"] == 2.0
        assert abs(mae_results["frequency"]["mae"] - 0.01) < 0.001

    def test_mae_integration_with_metrics_collector(self):
        """Test MAE integration with MetricsCollector."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create metrics collector with MAE enabled
            collector = MetricsCollector(log_directory=Path(tmp_dir), enable_mae_calculation=True)

            # Simulate prediction tracking
            base_time = datetime(2024, 1, 1, 12, 0, 0)

            # Add prediction
            collector.mae_calculator.add_prediction(
                base_time, {"power_generation": 150.0, "frequency": 60.02}, prediction_horizon_minutes=60
            )

            # Add actual
            collector.mae_calculator.add_actual(
                base_time + timedelta(hours=1), {"power_generation": 148.0, "frequency": 60.01}
            )

            # Get MAE metrics
            mae_metrics = collector.get_mae_metrics()

            assert "power_generation" in mae_metrics
            assert "frequency" in mae_metrics
            assert mae_metrics["power_generation"]["mae"] == 2.0


class TestMAEReporting:
    """Test MAE reporting and visualization functionality."""

    def test_mae_summary_report(self):
        """Test generation of MAE summary report."""
        calculator = MAECalculator()

        # Add sample data
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        for i in range(10):
            timestamp = base_time + timedelta(minutes=i * 15)

            calculator.add_prediction(
                timestamp, {"power_generation": 150.0 + i, "frequency": 60.0 + i * 0.01}, prediction_horizon_minutes=60
            )

            calculator.add_actual(
                timestamp + timedelta(hours=1), {"power_generation": 148.0 + i, "frequency": 59.99 + i * 0.01}
            )

        # Generate summary report
        summary = calculator.generate_summary_report()

        assert "power_generation" in summary
        assert "frequency" in summary
        assert "overall_accuracy" in summary
        assert "sample_count" in summary
        assert "time_period" in summary

        # Check that accuracy metrics are reasonable
        assert summary["power_generation"]["mae"] > 0
        assert summary["frequency"]["mae"] > 0
        assert 0 <= summary["overall_accuracy"] <= 100

    def test_mae_export_to_dataframe(self):
        """Test export of MAE data to pandas DataFrame."""
        calculator = MAECalculator()

        # Add sample data
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        for i in range(5):
            timestamp = base_time + timedelta(minutes=i * 15)

            calculator.add_prediction(timestamp, {"power_generation": 150.0 + i}, prediction_horizon_minutes=60)

            calculator.add_actual(timestamp + timedelta(hours=1), {"power_generation": 148.0 + i})

        # Export to DataFrame
        df = calculator.export_to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "prediction_timestamp" in df.columns
        assert "actual_timestamp" in df.columns
        assert "predicted_power_generation" in df.columns
        assert "actual_power_generation" in df.columns
        assert "error_power_generation" in df.columns

    def test_mae_performance_over_time(self):
        """Test MAE performance tracking over time."""
        calculator = MAECalculator()

        # Simulate 48 hours of predictions with changing accuracy
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        for hour in range(48):
            timestamp = base_time + timedelta(hours=hour)

            # Simulate daily cycle where predictions are worse at night
            time_of_day = hour % 24
            error_factor = 1.0 + 0.5 * np.sin(2 * np.pi * (time_of_day - 6) / 24)  # Peak error at midnight

            predicted_value = 100.0
            actual_value = predicted_value + error_factor * 5.0  # Variable error

            calculator.add_prediction(timestamp, {"power_generation": predicted_value}, prediction_horizon_minutes=60)

            calculator.add_actual(timestamp + timedelta(hours=1), {"power_generation": actual_value})

        # Calculate performance over time
        performance_timeline = calculator.calculate_performance_timeline(window_hours=6)

        # Should show variation in accuracy over time
        mae_values = [window["power_generation"]["mae"] for window in performance_timeline]
        assert max(mae_values) > min(mae_values)  # Should have variation
        assert len(performance_timeline) > 1  # Multiple time windows
