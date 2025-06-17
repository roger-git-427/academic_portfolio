# src/<your_package>/nodes.py

import os
import pickle
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.pytorch
import mlflow.prophet
from prophet import Prophet
import holidays
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from typing import Tuple


# ──────────────────────────────────────────────────────────────
#  Utility: plot helper
# ──────────────────────────────────────────────────────────────

def plot_forecast_vs_actual(y_true, y_pred, title="Forecast vs Actual", save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Forecast", linewidth=2)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def _compute_wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    return np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual))


# ──────────────────────────────────────────────────────────────
#  1️ Imputation & scaling 
# ──────────────────────────────────────────────────────────────

def impute_and_scale(df: pd.DataFrame, features: list[str]) -> Tuple[np.ndarray, list[str], StandardScaler]:
    """
    1) Impute missing lag_/rolling_ columns using Gaussian noise from last 7 observed rows
    2) Interpolate any remaining NaNs
    3) Scale all 'features' + 'target_habitaciones' using StandardScaler
    4) Save the scaler to disk (pickle)
    """
    print(f"[DEBUG][impute_and_scale] Input df shape: {df.shape}")
    print(f"[DEBUG][impute_and_scale] Features list: {features}")

    # Impute lag_/rolling_ columns via Gaussian noise
    for col in [c for c in features if c.startswith(("lag_", "rolling_"))]:
        np.random.seed(42)
        s = df[col].copy()
        for i in range(len(s)):
            if pd.isna(s.iat[i]):
                window = s[max(0, i - 7):i].dropna()
                if len(window) > 0:
                    s.iat[i] = np.random.normal(window.mean(), window.std())
                else:
                    s.iat[i] = 0.0
        df[col] = s
    print(f"[DEBUG][impute_and_scale] After imputation, subset df[features] shape: {df[features].shape}")

    # Interpolate any remaining NaNs
    df = df.interpolate()
    print(f"[DEBUG][impute_and_scale] After interpolation, subset df[features] shape: {df[features].shape}")

    # Fit‐transform scaler on features + target
    scaler = StandardScaler()
    arr = scaler.fit_transform(df[features + ["target_habitaciones"]])
    print(f"[DEBUG][impute_and_scale] Scaled array shape: {arr.shape}")

    # Save the scaler for inference later
    os.makedirs("data/06_models", exist_ok=True)
    scaler_path = "data/06_models/scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[DEBUG][impute_and_scale] Saved scaler to: {scaler_path}")

    return arr, features, scaler


def inverse_transform_target(scaler: StandardScaler, y_scaled: np.ndarray) -> np.ndarray:
    """
    Given a 1D array y_scaled, reconstruct a dummy array for inverse_transform,
    then extract the last column to get original‐scale y.
    """
    n_features = scaler.mean_.shape[0]
    dummy = np.zeros((len(y_scaled), n_features))
    dummy[:, -1] = y_scaled
    inversed = scaler.inverse_transform(dummy)[:, -1]
    return inversed


# ──────────────────────────────────────────────────────────────
#  2️ Create train/test split 
# ──────────────────────────────────────────────────────────────

def create_train_test(arr: np.ndarray, params: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    1) Turn `arr` into sliding windows of shape (window_size, num_features) and 1-step‐ahead y
    2) Perform a final TimeSeriesSplit(n_splits) to get train/test indices
    3) Return X_train, y_train, X_test, y_test as torch.FloatTensors
    """
    window_size = params["window_size"]
    horizon = params["horizon"]
    n_splits = params["n_splits"]
    print(f"[DEBUG][create_train_test] window_size={window_size}, horizon={horizon}, n_splits={n_splits}")

    class TSData(Dataset):
        def __init__(self, data: np.ndarray, w: int, h: int):
            X_list, y_list = [], []
            for i in range(len(data) - w - h + 1):
                X_list.append(data[i : i + w, :-1])
                y_list.append(data[i + w + h - 1, -1])
            self.X = torch.tensor(np.stack(X_list), dtype=torch.float32)
            self.y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    dataset = TSData(arr, window_size, horizon)
    splits = list(TimeSeriesSplit(n_splits=n_splits).split(dataset.X))
    train_idx, test_idx = splits[-1]

    X_train = dataset.X[train_idx]
    y_train = dataset.y[train_idx]
    X_test = dataset.X[test_idx]
    y_test = dataset.y[test_idx]

    print(f"[DEBUG][create_train_test] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[DEBUG][create_train_test] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test


# ──────────────────────────────────────────────────────────────
#  3️ SARIMA training 
# ──────────────────────────────────────────────────────────────

def train_sarima(df_features: pd.DataFrame, cfg: dict, scaler: StandardScaler) -> Tuple[float, float, float, str]:
    """
    1) Resample df_features to daily frequency, forward-fill missing
    2) Fit SARIMAX on train portion, forecast on test portion
    3) Log metrics (MAE, RMSE, WMAPE) under keys sarima_MAE, sarima_RMSE, sarima_WMAPE
    4) Save model pickle and MLflow artifact
    5) Plot forecast vs actual, log as artifact
    Returns mae, rmse, wmape, mlflow‐tracked model object, local pickle path
    """
    df_p = df_features.rename(columns={"Fecha": "ds", "target_habitaciones": "y"})
    df_p = df_p.set_index("ds").asfreq("D")
    df_p = df_p.fillna(method="ffill")

    series = df_p["y"]
    periods = cfg["periods"]
    train_series = series.iloc[:-periods]
    test_series = series.iloc[-periods :]

    print(f"[DEBUG][train_sarima] Training SARIMA(order={cfg['order']}, seasonal_order={cfg['seasonal_order']})")
    model = SARIMAX(
        train_series,
        order=tuple(cfg["order"]),
        seasonal_order=tuple(cfg["seasonal_order"]),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_result = model.fit(disp=False)

    forecast = sarima_result.forecast(steps=periods)
    y_true = test_series.values
    y_pred = forecast.values

    mae_val = mean_absolute_error(y_true, y_pred)
    rmse_val = mean_squared_error(y_true, y_pred) ** 0.5
    wmape_val = _compute_wmape(y_true, y_pred)

    # Log to MLflow
    with mlflow.start_run(run_name="SARIMA", nested=True):
        mlflow.log_params({
            "order": cfg["order"],
            "seasonal_order": cfg["seasonal_order"],
            "periods": cfg["periods"]
        })
        mlflow.log_metrics({
            "sarima_MAE": mae_val,
            "sarima_RMSE": rmse_val,
            "sarima_WMAPE": wmape_val
        })

        # Save SARIMA pickle
        os.makedirs("data/06_models", exist_ok=True)
        sarima_pickle_path = "data/06_models/sarima_model.pkl"
        with open(sarima_pickle_path, "wb") as f:
            pickle.dump(sarima_result, f)
        print(f"[DEBUG][train_sarima] Saved SARIMA pickle to {sarima_pickle_path}")

        # Log model via MLflow (statsmodels results cannot be logged via mlflow.pytorch,
        # so we only log the pickle as an artifact)
        mlflow.log_artifact(sarima_pickle_path, artifact_path="sarima_model")

        # Plot and log
        plot_path = "data/06_models/sarima_forecast.png"
        plot_forecast_vs_actual(y_true, y_pred, title="SARIMA Forecast vs Actual", save_path=plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")

    return mae_val, rmse_val, wmape_val, sarima_pickle_path


# ──────────────────────────────────────────────────────────────
#  4️ Transformer training 
# ──────────────────────────────────────────────────────────────

class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x_proj = self.input_proj(x)
        x_enc = self.transformer(x_proj)
        return self.fc(x_enc[:, -1, :])


def train_transformer(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: dict,
    scaler: StandardScaler,
) -> Tuple[float, float, float, nn.Module, str]:
    """
    1) Load best Optuna params if exist
    2) Train TransformerForecaster, evaluate on X_test
    3) Log metrics sarima_MAE -> transformer_MAE etc.
    4) Save model pickle and MLflow PyTorch model
    5) Plot forecast vs actual, log as artifact
    Returns mae, rmse, wmape, model object, local pickle path
    """
    import random

    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    model_type = "transformer"
    best_params_path = f"data/07_optuna/{model_type}_best_params.yml"
    if os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params = yaml.safe_load(f)
        cfg.update(best_params)
        cfg["epochs"] = 250
        print(f"[DEBUG][train_transformer] Using BEST params + epochs={cfg['epochs']}")
    else:
        print(f"[DEBUG][train_transformer] No best params found → using cfg from parameters.yml")

    # — Debug: print shapes and config —
    print(f"[DEBUG][train_transformer] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[DEBUG][train_transformer] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"[DEBUG][train_transformer] Transformer config: input_dim={cfg['input_dim']}, "
          f"d_model={cfg['d_model']}, nhead={cfg['nhead']}, num_layers={cfg['num_layers']}, "
          f"lr={cfg['lr']}, batch_size={cfg['batch_size']}, epochs={cfg['epochs']}")

        # — Instantiate model —
    model = TransformerForecaster(
            input_dim=cfg["input_dim"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
        ).to("cpu")

        # — Debug: print model input projection dims —
    expected_in = model.input_proj.in_features
    print(f"[DEBUG][train_transformer] Created TransformerForecaster with input_proj.in_features = {expected_in}, "
              f"output dimension = {model.fc.out_features}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    with mlflow.start_run(run_name="Transformer", nested=True):
        mlflow.log_params(cfg)

        # Training loop
        for epoch in range(cfg["epochs"]):
            model.train()
            for xb, yb in DataLoader(
                list(zip(X_train, y_train)), batch_size=cfg["batch_size"]
            ):
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(
                list(zip(X_test, y_test)), batch_size=cfg["batch_size"]
            ):
                p = model(xb).cpu().numpy().flatten()
                preds.extend(p)
                acts.extend(yb.cpu().numpy().flatten())

        preds = np.array(preds)
        acts = np.array(acts)
        preds_orig = inverse_transform_target(scaler, preds)
        acts_orig = inverse_transform_target(scaler, acts)

        mae_val = mean_absolute_error(acts_orig, preds_orig)
        rmse_val = mean_squared_error(acts_orig, preds_orig) ** 0.5
        wmape_val = _compute_wmape(acts_orig, preds_orig)
        print(f"[DEBUG][train_transformer] Evaluation MAE={mae_val:.4f}, RMSE={rmse_val:.4f}, WMAPE={wmape_val:.4f}")

        mlflow.log_metrics({
            "transformer_MAE": mae_val,
            "transformer_RMSE": rmse_val,
            "transformer_WMAPE": wmape_val
        })

        # Save pickle
        os.makedirs("data/06_models", exist_ok=True)
        transformer_pickle = "data/06_models/transformer_model.pkl"
        with open(transformer_pickle, "wb") as f:
            pickle.dump(model, f)
        print(f"[DEBUG][train_transformer] Saved Transformer pickle to {transformer_pickle}")
        mlflow.log_artifact(transformer_pickle, artifact_path="transformer_model")

        # Save MLflow PyTorch model
        mlflow.pytorch.log_model(model, artifact_path="transformer_model_pytorch")

        # Plot and log
        plot_path = "data/06_models/transformer_forecast.png"
        plot_forecast_vs_actual(acts_orig, preds_orig, title="Transformer Forecast vs Actual", save_path=plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")

    return mae_val, rmse_val, wmape_val, model, transformer_pickle


# ──────────────────────────────────────────────────────────────
#  5️ GRU training 
# ──────────────────────────────────────────────────────────────

class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def train_gru(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: dict,
    scaler: StandardScaler,
) -> Tuple[float, float, float, nn.Module, str]:
    """
    1) Load best Optuna params if exist
    2) Train GRUForecaster, evaluate
    3) Log metrics under keys gru_MAE, gru_RMSE, gru_WMAPE
    4) Save model pickle and MLflow PyTorch model
    5) Plot forecast vs actual, log as artifact
    Returns mae, rmse, wmape, model object, local pickle path
    """
    import random

    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    model_type = "gru"
    best_params_path = f"data/07_optuna/{model_type}_best_params.yml"
    if os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params = yaml.safe_load(f)
        cfg.update(best_params)
        cfg["epochs"] = 250
        print(f"[DEBUG][train_gru] Using BEST params + epochs={cfg['epochs']}")
    else:
        print(f"[DEBUG][train_gru] No best params found → using cfg from parameters.yml")

    print(f"[DEBUG][train_gru] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[DEBUG][train_gru] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"[DEBUG][train_gru] GRU config: input_dim={cfg['input_dim']}, hidden_dim={cfg['hidden_dim']}, "
          f"num_layers={cfg['num_layers']}, lr={cfg['lr']}, batch_size={cfg['batch_size']}, "
          f"epochs={cfg['epochs']}")

    model = GRUForecaster(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to("cpu")

    # — Debug: Print the GRU’s expected input dimension (for sanity) —
    expected_in = model.gru.input_size if hasattr(model.gru, "input_size") else cfg["input_dim"]
    print(f"[DEBUG][train_gru] Created GRUForecaster with input_size = {expected_in}, "
              f"hidden_dim = {cfg['hidden_dim']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    with mlflow.start_run(run_name="GRU", nested=True):
        mlflow.log_params(cfg)

        # Training loop
        for epoch in range(cfg["epochs"]):
            model.train()
            for xb, yb in DataLoader(
                list(zip(X_train, y_train)), batch_size=cfg["batch_size"]
            ):
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(
                list(zip(X_test, y_test)), batch_size=cfg["batch_size"]
            ):
                p = model(xb).cpu().numpy().flatten()
                preds.extend(p)
                acts.extend(yb.cpu().numpy().flatten())

        preds = np.array(preds)
        acts = np.array(acts)
        preds_orig = inverse_transform_target(scaler, preds)
        acts_orig = inverse_transform_target(scaler, acts)

        mae_val = mean_absolute_error(acts_orig, preds_orig)
        rmse_val = mean_squared_error(acts_orig, preds_orig) ** 0.5
        wmape_val = _compute_wmape(acts_orig, preds_orig)
        print(f"[DEBUG][train_gru] Evaluation MAE={mae_val:.4f}, RMSE={rmse_val:.4f}, WMAPE={wmape_val:.4f}")

        mlflow.log_metrics({
            "gru_MAE": mae_val,
            "gru_RMSE": rmse_val,
            "gru_WMAPE": wmape_val
        })

        # Save pickle
        os.makedirs("data/06_models", exist_ok=True)
        gru_pickle = "data/06_models/gru_model.pkl"
        with open(gru_pickle, "wb") as f:
            pickle.dump(model, f)
        print(f"[DEBUG][train_gru] Saved GRU pickle to {gru_pickle}")
        mlflow.log_artifact(gru_pickle, artifact_path="gru_model")

        # Save MLflow PyTorch model
        mlflow.pytorch.log_model(model, artifact_path="gru_model_pytorch")

        # Plot and log
        plot_path = "data/06_models/gru_forecast.png"
        plot_forecast_vs_actual(acts_orig, preds_orig, title="GRU Forecast vs Actual", save_path=plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")

    return mae_val, rmse_val, wmape_val, model, gru_pickle


# ──────────────────────────────────────────────────────────────
#  6️ LSTM training 
# ──────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: dict,
    scaler: StandardScaler,
) -> Tuple[float, float, float, nn.Module, str]:
    """
    1) Load best Optuna params if exist
    2) Train LSTMForecaster, evaluate
    3) Log metrics under keys lstm_MAE, lstm_RMSE, lstm_WMAPE
    4) Save model pickle and MLflow PyTorch model
    5) Plot forecast vs actual, log as artifact
    Returns mae, rmse, wmape, model object, local pickle path
    """
    import random

    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    model_type = "lstm"
    best_params_path = f"data/07_optuna/{model_type}_best_params.yml"
    if os.path.exists(best_params_path):
        print(f"✅ Loading best params from {best_params_path}")
        with open(best_params_path, "r") as f:
            best_params = yaml.safe_load(f)
        cfg.update(best_params)
        cfg["epochs"] = cfg.get("epochs", 250)
        print(f"Using BEST params + epochs={cfg['epochs']}")
    else:
        print(f"⚠️ No best params found → using cfg from parameters.yml")

    # Ensure input_dim matches actual X_train
    cfg["input_dim"] = X_train.shape[-1]
    model = LSTMForecaster(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    with mlflow.start_run(run_name="LSTM", nested=True):
        mlflow.log_params(cfg)

        # Training loop
        for epoch in range(cfg["epochs"]):
            model.train()
            for xb, yb in DataLoader(
                list(zip(X_train, y_train)), batch_size=cfg["batch_size"]
            ):
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(
                list(zip(X_test, y_test)), batch_size=cfg["batch_size"]
            ):
                p = model(xb).cpu().numpy().flatten()
                preds.extend(p)
                acts.extend(yb.cpu().numpy().flatten())

        preds = np.array(preds)
        acts = np.array(acts)
        preds_orig = inverse_transform_target(scaler, preds)
        acts_orig = inverse_transform_target(scaler, acts)

        mae_val = mean_absolute_error(acts_orig, preds_orig)
        rmse_val = mean_squared_error(acts_orig, preds_orig) ** 0.5
        wmape_val = _compute_wmape(acts_orig, preds_orig)

        mlflow.log_metrics({
            "lstm_MAE": mae_val,
            "lstm_RMSE": rmse_val,
            "lstm_WMAPE": wmape_val
        })

        # Save pickle
        os.makedirs("data/06_models", exist_ok=True)
        lstm_pickle = "data/06_models/lstm_model.pkl"
        with open(lstm_pickle, "wb") as f:
            pickle.dump(model, f)
        print(f"[DEBUG][train_lstm] Saved LSTM pickle to {lstm_pickle}")
        mlflow.log_artifact(lstm_pickle, artifact_path="lstm_model")

        # Save MLflow PyTorch model
        mlflow.pytorch.log_model(model, artifact_path="lstm_model_pytorch")

        # Plot and log
        plot_path = "data/06_models/lstm_forecast.png"
        plot_forecast_vs_actual(acts_orig, preds_orig, title="LSTM Forecast vs Actual", save_path=plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")

    return mae_val, rmse_val, wmape_val, model, lstm_pickle


# ──────────────────────────────────────────────────────────────
#  7️ Prophet training 
# ──────────────────────────────────────────────────────────────

def train_prophet(
    df_features: pd.DataFrame, cfg: dict, scaler: StandardScaler
) -> Tuple[float, float, float, Prophet, str]:
    """
    1) Prepare ds/y and regressors for Prophet
    2) Fit Prophet on train, predict on test
    3) Log metrics under prophet_MAE, prophet_RMSE, prophet_WMAPE
    4) Save model pickle and MLflow Prophet model
    5) Plot forecast vs actual and components, log as artifacts
    Returns mae, rmse, wmape, model object, local pickle path
    """
    print(f"[DEBUG][train_prophet] Input shape: {df_features.shape}")
    print(f"[DEBUG][train_prophet] Prophet periods: {cfg['periods']}")

    with mlflow.start_run(run_name="Prophet", nested=True):
        mlflow.log_params(cfg)

        df_p = df_features.rename(columns={"Fecha": "ds", "target_habitaciones": "y"})
        df_p["is_weekend"] = df_p["ds"].dt.weekday.isin([5, 6]).astype(int)
        mx = holidays.Mexico()
        df_p["is_holiday"] = df_p["ds"].isin(mx).astype(int)

        regressors = [
            "is_weekend",
            "is_holiday",
            "lag_7",
            "lag_30",
            "rolling_mean_7",
            "rolling_mean_21",
            "rolling_std_30",
        ]
        print(f"[DEBUG][train_prophet] Regressors: {regressors}")

        df_train = df_p.iloc[:-cfg["periods"]].dropna(subset=regressors + ["y", "ds"]).copy()
        df_test = df_p.iloc[-cfg["periods"] :].dropna(subset=regressors + ["y", "ds"]).copy()

        print(f"[DEBUG][train_prophet] df_train shape (after dropna): {df_train.shape}")
        print(f"[DEBUG][train_prophet] df_test shape (after dropna): {df_test.shape}")

        # Initialize and add regressors
        m = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        for reg in regressors:
            m.add_regressor(reg)

        m.fit(df_train)
        print(f"[DEBUG][train_prophet] Fitted Prophet model")

        future = df_test[["ds"] + regressors]
        forecast = m.predict(future)
        print(f"[DEBUG][train_prophet] Completed Prophet.predict(), forecast shape: {forecast.shape}")

        # Compute metrics
        y_true = df_test["y"].values
        y_pred = forecast["yhat"].values

        mae_val = mean_absolute_error(y_true, y_pred)
        rmse_val = mean_squared_error(y_true, y_pred) ** 0.5
        wmape_val = _compute_wmape(y_true, y_pred)
        print(f"[DEBUG][train_prophet] Evaluation MAE={mae_val:.4f}, RMSE={rmse_val:.4f}, WMAPE={wmape_val:.4f}")

        mlflow.log_metrics({
            "prophet_MAE": mae_val,
            "prophet_RMSE": rmse_val,
            "prophet_WMAPE": wmape_val
        })

        # Save pickle
        os.makedirs("data/06_models", exist_ok=True)
        prophet_pickle = "data/06_models/prophet_model.pkl"
        with open(prophet_pickle, "wb") as f:
            pickle.dump(m, f)
        print(f"[DEBUG][train_prophet] Saved Prophet pickle to {prophet_pickle}")
        mlflow.log_artifact(prophet_pickle, artifact_path="prophet_model")

        # Save MLflow Prophet model
        mlflow.prophet.log_model(m, artifact_path="prophet_model_pytorch")

        # Plot forecast and components
        plot_path = "data/06_models/prophet_forecast.png"
        plot_forecast_vs_actual(y_true, y_pred, title="Prophet Forecast vs Actual", save_path=plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")

        components_fig = m.plot_components(forecast)
        comp_path = "data/06_models/prophet_components.png"
        components_fig.savefig(comp_path)
        mlflow.log_artifact(comp_path, artifact_path="plots")

    return mae_val, rmse_val, wmape_val, m, prophet_pickle
