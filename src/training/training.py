import json
import os
import pathlib
from dataclasses import dataclass
from typing import Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn import ensemble
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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

    if not is_classification:
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

        if step is None or (step + 1) % 10 == 0:
            print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
            print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
            print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

    # Calculate and log classification metrics if applicable
    else:
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


def train_model(data: Data):
    raw_params = os.environ.get("PARAMS")
    if raw_params is None:
        raise ValueError("PARAMS environment variable is not set")

    raw_model = os.environ.get("MODEL")
    if raw_model is None:
        raise ValueError("MODEL environment variable is not set")

    is_classification = bool(int(os.environ.get("IS_CLASSIFICATION", "0")))
    params: dict = json.loads(raw_params)

    # Encode labels if classification
    y_train = data.y_train
    y_test = data.y_test
    label_encoder = None

    if is_classification:
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(data.y_train)
        y_test = label_encoder.transform(data.y_test)
        # Create new Data object with encoded labels
        data = Data(
            X_train=data.X_train,
            X_test=data.X_test,
            y_train=pd.Series(y_train),
            y_test=pd.Series(y_test),
        )

    label_mapping = None
    if is_classification and label_encoder is not None:
         label_mapping = {int(i): str(label) for i, label in enumerate(label_encoder.classes_)}

    if raw_model == "RandomForestClassifier":
        model = ensemble.RandomForestClassifier(**params)
        train_generic_model(model=model, data=data, params=params, label_mapping=label_mapping)
    elif raw_model == "RandomForestRegressor":
        model = ensemble.RandomForestRegressor(**params)
        train_generic_model(model=model, data=data, params=params, label_mapping=label_mapping)
    elif raw_model == "Xgboost":
        model = xgb.XGBClassifier(**params)
        train_generic_model(model=model, data=data, params=params, label_mapping=label_mapping)
    elif raw_model == "XgboostRegressor":
        model = xgb.XGBRegressor(**params)
        train_generic_model(model=model, data=data, params=params, label_mapping=label_mapping)
    elif raw_model == "Linear":
        params["in_features"] = data.X_train.shape[1]
        if is_classification:
            params["out_features"] = len(np.unique(data.y_train))
        else:
            params["out_features"] = 1

        model = nn.Linear(**params)
        train_torch_model(
            model=model, data=data, params=params, is_classification=is_classification, label_mapping=label_mapping
        )
    elif raw_model == "Sequential":
        # Dynamically determine input_dim from the data
        input_dim = data.X_train.shape[1]
        params["input_dim"] = input_dim
        if is_classification:
            params["output_dim"] = len(np.unique(data.y_train))
        else:
            params["output_dim"] = 1

        model = nn.Sequential(
            nn.Linear(
                in_features=params["input_dim"], out_features=params["hidden_dim"]
            ),
            *(
                [
                    nn.ReLU(inplace=True),
                    nn.Linear(
                        in_features=params["hidden_dim"],
                        out_features=params["hidden_dim"],
                    ),
                ]
                * params["layers"]
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=params["hidden_dim"], out_features=params["output_dim"]
            ),
            # Remove Softmax for CrossEntropyLoss
        )
        train_torch_model(
            model=model, data=data, params=params, is_classification=is_classification, label_mapping=label_mapping
        )
    else:
        raise ValueError(f"Unhandled model type: {raw_model}.")


def train_generic_model(
    model: Union[BaseEstimator, xgb.XGBClassifier], data: Data, params: dict, label_mapping: dict = None
):
    """Train sklearn models (RandomForest, XGBoost, etc.)"""
    is_classification = bool(int(os.environ.get("IS_CLASSIFICATION", "0")))
    model_name = os.environ.get("MODEL", "Unknown")
    dataset_name = os.environ.get("DATASET_FILENAME", "dataset")
    mlflow_experiment_name = str(
        os.environ.get("MLFLOW_EXPERIMENT", "automl-experiments")
    )

    experiment = mlflow.set_experiment(mlflow_experiment_name)

    run_name = f"{model_name}_{dataset_name}"

    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", dataset_name)
        mlflow.set_tag(
            "task_type", "classification" if is_classification else "regression"
        )
        mlflow.set_tag("framework", "sklearn")
        for name, value in params.items():
            mlflow.log_param(name, value)
        # Save label mapping if provided
        if label_mapping:
             with open("labels.json", "w") as f:
                 json.dump(label_mapping, f)
             mlflow.log_artifact("labels.json", "model")
             os.remove("labels.json")

        model.fit(data.X_train, data.y_train)

        train_predictions = model.predict(data.X_train)
        test_predictions = model.predict(data.X_test)

        calculate_and_log_metrics(
            y_train_true=data.y_train,
            y_train_pred=train_predictions,
            y_test_true=data.y_test,
            y_test_pred=test_predictions,
            is_classification=is_classification,
        )

        # Log the model
        registered_name = f"{pathlib.Path(dataset_name).stem}_{model_name}"
        mlflow.sklearn.log_model(model, "model", registered_model_name=registered_name)


def train_torch_model(
    model: torch.nn.Module, data: Data, params: dict, is_classification: bool = False, label_mapping: dict = None
):
    num_epochs = int(os.environ.get("NUM_EPOCHS", "100"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.001"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    model_name = os.environ.get("MODEL", "Unknown")
    dataset_name = os.environ.get("DATASET_FILENAME", "dataset")

    if is_classification:
        y_train_tensor = torch.LongTensor(data.y_train.values)
        y_test_tensor = torch.LongTensor(data.y_test.values)
    else:
        y_train_tensor = torch.FloatTensor(data.y_train.values).reshape(-1, 1)
        y_test_tensor = torch.FloatTensor(data.y_test.values).reshape(-1, 1)

    X_train_tensor = torch.FloatTensor(data.X_train.values)
    X_test_tensor = torch.FloatTensor(data.X_test.values)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    if is_classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlflow_experiment_name = str(
        os.environ.get("MLFLOW_EXPERIMENT", "automl-experiments")
    )
    experiment = mlflow.set_experiment(mlflow_experiment_name)

    run_name = f"{model_name}_{dataset_name}"

    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", dataset_name)
        mlflow.set_tag(
            "task_type", "classification" if is_classification else "regression"
        )
        mlflow.set_tag("framework", "pytorch")
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        for name, value in params.items():
            mlflow.log_param(name, value)
            
        # Save label mapping if provided
        if label_mapping:
             with open("labels.json", "w") as f:
                 json.dump(label_mapping, f)
             mlflow.log_artifact("labels.json", "model")
             os.remove("labels.json")


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

                if is_classification:
                    _, train_pred_classes = torch.max(train_predictions, 1)
                    _, test_pred_classes = torch.max(test_predictions, 1)
                    y_train_np = y_train_tensor.numpy()
                    train_pred_np = train_pred_classes.numpy()
                    y_test_np = y_test_tensor.numpy()
                    test_pred_np = test_pred_classes.numpy()
                else:
                    y_train_np = y_train_tensor.numpy().flatten()
                    train_pred_np = train_predictions.numpy().flatten()
                    y_test_np = y_test_tensor.numpy().flatten()
                    test_pred_np = test_predictions.numpy().flatten()

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)

            calculate_and_log_metrics(
                y_train_true=y_train_np,
                y_train_pred=train_pred_np,
                y_test_true=y_test_np,
                y_test_pred=test_pred_np,
                is_classification=is_classification,
                step=epoch,
            )

        registered_name = f"{pathlib.Path(dataset_name).stem}_{model_name}"
        mlflow.pytorch.log_model(model, "model", registered_model_name=registered_name)


def main():
    df = load_dataset()
    data = split_dataset(df)

    train_model(data)


if __name__ == "__main__":
    main()
