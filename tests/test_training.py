import json
from unittest.mock import patch

import pytest
import pandas as pd
import numpy as np

from training.training import (
    Data,
    load_dataset,
    main,
    split_dataset,
    train_model,
)


@pytest.fixture
def dummy_data(tmp_path):
    """Creates a temporary CSV file with dummy data."""
    df = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "target": np.random.rand(100),
        }
    )
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def _mock_env(monkeypatch, dummy_data):
    """Sets up required environment variables."""
    monkeypatch.setenv("DATA_PATH", dummy_data)
    monkeypatch.setenv("TARGET", "target")
    monkeypatch.setenv("MODEL", "Linear")
    # Params must match the __init__ of SimpleLinearModel
    params = json.dumps({"in_features": 2, "out_features": 1})
    monkeypatch.setenv("PARAMS", params)
    monkeypatch.setenv("NUM_EPOCHS", "1")  # Keep it fast
    monkeypatch.setenv("BATCH_SIZE", "10")
    monkeypatch.setenv("IS_CLASSIFICATION", "0")


@pytest.fixture
def mock_mlflow():
    """Mocks MLflow to prevent actual logging during tests."""
    with patch("training.training.mlflow") as mock:
        yield mock


def test_load_dataset(_mock_env):
    """Test if dataset loads correctly."""
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "feature1" in df.columns


def test_load_dataset_invalid_path(monkeypatch):
    """Test error handling for missing file."""
    monkeypatch.setenv("DATA_PATH", "non_existent.csv")
    with pytest.raises(FileNotFoundError):
        load_dataset()


def test_split_dataset(_mock_env):
    """Test data splitting logic."""
    df = load_dataset()
    data = split_dataset(df)

    assert isinstance(data, Data)
    # Default split is 0.2, so train should be ~80% of 100 rows
    assert len(data.X_train) == 80
    assert len(data.X_test) == 20
    # Ensure target column is removed from X
    assert "target" not in data.X_train.columns


def test_train_model_regression(_mock_env, mock_mlflow):
    """Test the training loop and MLflow logging calls (Regression)."""
    df = load_dataset()
    data = split_dataset(df)

    train_model(data)

    # Verify MLflow interactions
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_param.assert_any_call("learning_rate", 0.001)

    # Check if metrics were logged
    # We expect test_mae, test_rmse, etc.
    args_list = mock_mlflow.log_metric.call_args_list
    metric_names = [args[0][0] for args in args_list]
    assert "test_loss" in metric_names
    assert "test_mae" in metric_names


def test_train_model_classification_flag(_mock_env, mock_mlflow, monkeypatch):
    """Test if classification logic triggers when flag is set."""
    monkeypatch.setenv("IS_CLASSIFICATION", "1")

    df = load_dataset()
    data = split_dataset(df)

    train_model(data)

    # Check if accuracy was logged (only happens in classification branch)
    args_list = mock_mlflow.log_metric.call_args_list
    metric_names = [args[0][0] for args in args_list]
    assert "test_accuracy" in metric_names


def test_main_integration(_mock_env, mock_mlflow):
    """Run the whole training function."""
    # This ensures the glue code in main() works
    main()
    mock_mlflow.start_run.assert_called_once()


def test_train_model_with_sequential_layers(monkeypatch, dummy_data, mock_mlflow):
    """Test training with Sequential model using 'layers' parameter."""
    monkeypatch.setenv("DATA_PATH", dummy_data)
    monkeypatch.setenv("TARGET", "target")
    monkeypatch.setenv("MODEL", "Sequential")

    params = json.dumps(
        {
            "input_dim": 2,
            "hidden_dim": 16,
            "layers": 3,
            "output_dim": 1,
        }
    )
    monkeypatch.setenv("PARAMS", params)
    monkeypatch.setenv("NUM_EPOCHS", "1")
    monkeypatch.setenv("BATCH_SIZE", "10")
    monkeypatch.setenv("IS_CLASSIFICATION", "0")

    df = load_dataset()
    data = split_dataset(df)

    train_model(data)

    # Verify MLflow interactions
    mock_mlflow.start_run.assert_called_once()

    # Verify the model-specific params were logged
    mock_mlflow.log_param.assert_any_call("input_dim", 2)
    mock_mlflow.log_param.assert_any_call("hidden_dim", 16)
    mock_mlflow.log_param.assert_any_call("layers", 3)
    mock_mlflow.log_param.assert_any_call("output_dim", 1)

    # Verify metrics were logged
    args_list = mock_mlflow.log_metric.call_args_list
    metric_names = [args[0][0] for args in args_list]
    assert "test_loss" in metric_names
    assert "test_mae" in metric_names
    assert "test_rmse" in metric_names
    assert "test_r2" in metric_names


def test_train_model_custom_vs_sequential(monkeypatch, dummy_data, mock_mlflow):
    """Test that model creation differs based on presence of 'layers' parameter."""
    monkeypatch.setenv("DATA_PATH", dummy_data)
    monkeypatch.setenv("TARGET", "target")
    monkeypatch.setenv("NUM_EPOCHS", "1")
    monkeypatch.setenv("BATCH_SIZE", "10")
    monkeypatch.setenv("IS_CLASSIFICATION", "0")

    monkeypatch.setenv("MODEL", "Linear")
    params_custom = json.dumps({"in_features": 2, "out_features": 1})
    monkeypatch.setenv("PARAMS", params_custom)

    df = load_dataset()
    data = split_dataset(df)
    train_model(data)

    assert mock_mlflow.start_run.call_count == 1

    monkeypatch.setenv("MODEL", "Sequential")
    params_sequential = json.dumps(
        {"input_dim": 2, "hidden_dim": 8, "layers": 2, "output_dim": 1}
    )
    monkeypatch.setenv("PARAMS", params_sequential)

    train_model(data)

    assert mock_mlflow.start_run.call_count == 2


def test_train_sklearn_random_forest(monkeypatch, dummy_data, mock_mlflow):
    """Test training with RandomForestRegressor."""
    monkeypatch.setenv("DATA_PATH", dummy_data)
    monkeypatch.setenv("TARGET", "target")
    monkeypatch.setenv("MODEL", "RandomForestRegressor")

    params = json.dumps({"n_estimators": 10, "max_depth": 3, "random_state": 42})
    monkeypatch.setenv("PARAMS", params)
    monkeypatch.setenv("IS_CLASSIFICATION", "0")

    df = load_dataset()
    data = split_dataset(df)
    train_model(data)

    # Verify MLflow interactions
    mock_mlflow.start_run.assert_called_once()

    # Verify sklearn-specific params were logged
    mock_mlflow.log_param.assert_any_call("n_estimators", 10)
    mock_mlflow.log_param.assert_any_call("max_depth", 3)

    # Verify sklearn metrics were logged (no epochs, just final metrics)
    args_list = mock_mlflow.log_metric.call_args_list
    metric_names = [args[0][0] for args in args_list]
    assert "train_mae" in metric_names
    assert "test_mae" in metric_names
    assert "train_rmse" in metric_names
    assert "test_rmse" in metric_names
    assert "train_r2" in metric_names
    assert "test_r2" in metric_names

    # Verify sklearn model logging
    mock_mlflow.sklearn.log_model.assert_called_once()


def test_train_sklearn_xgboost(monkeypatch, dummy_data, mock_mlflow):
    """Test training with XGBoost Regressor."""
    monkeypatch.setenv("DATA_PATH", dummy_data)
    monkeypatch.setenv("TARGET", "target")
    monkeypatch.setenv("MODEL", "XgboostRegressor")

    params = json.dumps({"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1})
    monkeypatch.setenv("PARAMS", params)
    monkeypatch.setenv("IS_CLASSIFICATION", "0")

    df = load_dataset()
    data = split_dataset(df)
    train_model(data)

    # Verify MLflow interactions
    mock_mlflow.start_run.assert_called_once()

    # Verify xgboost-specific params were logged
    mock_mlflow.log_param.assert_any_call("n_estimators", 10)
    mock_mlflow.log_param.assert_any_call("learning_rate", 0.1)

    # Verify metrics
    args_list = mock_mlflow.log_metric.call_args_list
    metric_names = [args[0][0] for args in args_list]
    assert "test_mae" in metric_names
    assert "test_r2" in metric_names

    # Verify sklearn model logging (XGBoost uses sklearn interface)
    mock_mlflow.sklearn.log_model.assert_called_once()


def test_train_sklearn_with_classification(monkeypatch, dummy_data, mock_mlflow):
    """Test sklearn model with classification metrics."""
    # Create binary classification data
    df = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "target": np.random.randint(0, 2, 100),  # Binary classification
        }
    )
    file_path = dummy_data.replace("data.csv", "classification_data.csv")
    df.to_csv(file_path, index=False)

    monkeypatch.setenv("DATA_PATH", file_path)
    monkeypatch.setenv("TARGET", "target")
    monkeypatch.setenv("MODEL", "RandomForestClassifier")
    monkeypatch.setenv("PARAMS", json.dumps({"n_estimators": 10, "random_state": 42}))
    monkeypatch.setenv("IS_CLASSIFICATION", "1")

    df = load_dataset()
    data = split_dataset(df)
    train_model(data)

    # Verify classification metrics were logged
    args_list = mock_mlflow.log_metric.call_args_list
    metric_names = [args[0][0] for args in args_list]
    assert "train_accuracy" in metric_names
    assert "test_accuracy" in metric_names
    assert "train_f1" in metric_names
    assert "test_f1" in metric_names


def test_train_torch_vs_sklearn_different_metrics(monkeypatch, dummy_data, mock_mlflow):
    """Test that torch and sklearn models log metrics differently."""
    monkeypatch.setenv("DATA_PATH", dummy_data)
    monkeypatch.setenv("TARGET", "target")
    monkeypatch.setenv("NUM_EPOCHS", "2")
    monkeypatch.setenv("BATCH_SIZE", "10")
    monkeypatch.setenv("IS_CLASSIFICATION", "0")

    df = load_dataset()
    data = split_dataset(df)

    # Test torch model - should log metrics per epoch with step parameter
    monkeypatch.setenv("MODEL", "Linear")
    monkeypatch.setenv("PARAMS", json.dumps({"in_features": 2, "out_features": 1}))
    train_model(data)

    torch_calls = mock_mlflow.log_metric.call_args_list
    # Check if any call has 'step' keyword argument (torch models log per epoch)
    has_step = any("step" in str(call) for call in torch_calls)
    assert has_step, "Torch model should log metrics with step parameter"

    mock_mlflow.reset_mock()

    # Test sklearn model - should log metrics once without step parameter
    monkeypatch.setenv("MODEL", "RandomForestRegressor")
    monkeypatch.setenv("PARAMS", json.dumps({"n_estimators": 10}))
    train_model(data)

    sklearn_calls = mock_mlflow.log_metric.call_args_list
    # sklearn models should not have step parameter
    sklearn_metric_names = [args[0][0] for args in sklearn_calls]
    assert "train_mae" in sklearn_metric_names
    assert "test_mae" in sklearn_metric_names


def test_invalid_model_name(monkeypatch, dummy_data):
    """Test that invalid model names raise appropriate errors."""
    monkeypatch.setenv("DATA_PATH", dummy_data)
    monkeypatch.setenv("TARGET", "target")
    monkeypatch.setenv("MODEL", "InvalidModelName")
    monkeypatch.setenv("PARAMS", json.dumps({}))

    df = load_dataset()
    data = split_dataset(df)

    with pytest.raises(ValueError, match="Unhandled model type"):
        train_model(data)
