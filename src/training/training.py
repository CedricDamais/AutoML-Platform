from dataclasses import dataclass
from typing import Tuple, Union
import json
import os
import pathlib

from sklearn import ensemble
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    accuracy_score,
    f1_score,
)
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb


@dataclass
class Data:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


def load_dataset(**kwargs) -> pd.DataFrame:
    path = os.environ.get("DATA_PATH")

    if path is None:
        raise ValueError("DATA_PATH environment variable is not set")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    ext = pathlib.Path(path).suffix.lower()

    loaders = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
        ".pkl": pd.read_pickle,
        ".pickle": pd.read_pickle,
        ".feather": pd.read_feather,
        ".xml": pd.read_xml,
        ".h5": pd.read_hdf,
        ".hdf5": pd.read_hdf,
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported file format: {ext}")

    return loaders[ext](path, **kwargs)


def split_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = os.environ.get("TARGET")
    if target is None:
        raise ValueError("TARGET environment variable is not set")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return Data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def calculate_and_log_metrics(
    y_train_true,
    y_train_pred,
    y_test_true,
    y_test_pred,
    is_classification: bool,
    step: int = None,
):
    """Calculate and log both regression and classification metrics."""
    # Calculate regression metrics
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    train_r2 = r2_score(y_train_true, y_train_pred)
    test_r2 = r2_score(y_test_true, y_test_pred)

    # Log regression metrics
    mlflow.log_metric("train_mae", train_mae, step=step)
    mlflow.log_metric("test_mae", test_mae, step=step)
    mlflow.log_metric("train_rmse", train_rmse, step=step)
    mlflow.log_metric("test_rmse", test_rmse, step=step)
    mlflow.log_metric("train_r2", train_r2, step=step)
    mlflow.log_metric("test_r2", test_r2, step=step)

    # Calculate and log classification metrics if applicable
    if is_classification:
        # Convert continuous predictions to class labels
        y_train_pred_class = np.round(y_train_pred).astype(int)
        y_test_pred_class = np.round(y_test_pred).astype(int)
        y_train_true_class = np.asarray(y_train_true).astype(int)
        y_test_true_class = np.asarray(y_test_true).astype(int)

        train_accuracy = accuracy_score(y_train_true_class, y_train_pred_class)
        test_accuracy = accuracy_score(y_test_true_class, y_test_pred_class)
        train_f1 = f1_score(
            y_train_true_class, y_train_pred_class, average="weighted", zero_division=0
        )
        test_f1 = f1_score(
            y_test_true_class, y_test_pred_class, average="weighted", zero_division=0
        )

        mlflow.log_metric("train_accuracy", train_accuracy, step=step)
        mlflow.log_metric("test_accuracy", test_accuracy, step=step)
        mlflow.log_metric("train_f1", train_f1, step=step)
        mlflow.log_metric("test_f1", test_f1, step=step)

        if step is None or (step + 1) % 10 == 0:
            print(
                f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
            )
            print(f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
    else:
        if step is None or (step + 1) % 10 == 0:
            print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
            print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
            print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")


def train_model(data: Data):
    raw_params = os.environ.get("PARAMS")
    if raw_params is None:
        raise ValueError("PARAMS environment variable is not set")

    raw_model = os.environ.get("MODEL")
    if raw_model is None:
        raise ValueError("MODEL environment variable is not set")

    params: dict = json.loads(raw_params)
    if raw_model == "RandomForestClassifier":
        model = ensemble.RandomForestClassifier(**params)
        train_generic_model(model=model, data=data, params=params)
    elif raw_model == "RandomForestRegressor":
        model = ensemble.RandomForestRegressor(**params)
        train_generic_model(model=model, data=data, params=params)
    elif raw_model == "Xgboost":
        model = xgb.XGBClassifier(**params)
        train_generic_model(model=model, data=data, params=params)
    elif raw_model == "XgboostRegressor":
        model = xgb.XGBRegressor(**params)
        train_generic_model(model=model, data=data, params=params)
    elif raw_model == "Linear":
        model = nn.Linear(**params)
        train_torch_model(model=model, data=data, params=params)
    elif raw_model == "Sequential":
        model = nn.Sequential(
            nn.Linear(
                in_features=params["input_dim"], out_features=params["hidden_dim"]
            ),
            *(
                [
                    nn.Linear(
                        in_features=params["hidden_dim"],
                        out_features=params["hidden_dim"],
                    ),
                ]
                * params["layers"]
            ),
            nn.Linear(
                in_features=params["hidden_dim"], out_features=params["output_dim"]
            ),
        )
        train_torch_model(model=model, data=data, params=params)
    else:
        raise ValueError(f"Unhandled model type: {raw_model}.")


def train_generic_model(
    model: Union[BaseEstimator, xgb.XGBClassifier], data: Data, params: dict
):
    """Train sklearn models (RandomForest, XGBoost, etc.)"""
    is_classification = bool(int(os.environ.get("IS_CLASSIFICATION", "0")))

    mlflow.set_tracking_uri(uri="http://0.0.0.0:8080")

    with mlflow.start_run():
        # Log parameters
        for name, value in params.items():
            mlflow.log_param(name, value)

        # Train the model
        model.fit(data.X_train, data.y_train)

        # Make predictions
        train_predictions = model.predict(data.X_train)
        test_predictions = model.predict(data.X_test)

        # Calculate and log all metrics
        calculate_and_log_metrics(
            y_train_true=data.y_train,
            y_train_pred=train_predictions,
            y_test_true=data.y_test,
            y_test_pred=test_predictions,
            is_classification=is_classification,
        )

        # Log the model
        mlflow.sklearn.log_model(model, "model")


def train_torch_model(model: torch.nn.Module, data: Data, params: dict):
    num_epochs = int(os.environ.get("NUM_EPOCHS", "100"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.001"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    is_classification = bool(os.environ.get("IS_CLASSIFICATION", 0))

    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(data.X_train.values)
    y_train_tensor = torch.FloatTensor(data.y_train.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(data.X_test.values)
    y_test_tensor = torch.FloatTensor(data.y_test.values).reshape(-1, 1)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_tracking_uri(uri="http://0.0.0.0:8080")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        for name, value in params.items():
            mlflow.log_param(name, value)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            model.eval()
            with torch.no_grad():
                train_predictions = model(X_train_tensor)
                test_predictions = model(X_test_tensor)
                test_loss = criterion(test_predictions, y_test_tensor).item()

                # Convert to numpy for sklearn metrics
                y_train_np = y_train_tensor.numpy().flatten()
                train_pred_np = train_predictions.numpy().flatten()
                y_test_np = y_test_tensor.numpy().flatten()
                test_pred_np = test_predictions.numpy().flatten()

            # Log training loss
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)

            # Calculate and log all metrics
            calculate_and_log_metrics(
                y_train_true=y_train_np,
                y_train_pred=train_pred_np,
                y_test_true=y_test_np,
                y_test_pred=test_pred_np,
                is_classification=is_classification,
                step=epoch,
            )

        # Log the model
        mlflow.pytorch.log_model(model, "model")


def main():
    df = load_dataset()
    data = split_dataset(df)

    train_model(data)


if __name__ == "__main__":
    main()
