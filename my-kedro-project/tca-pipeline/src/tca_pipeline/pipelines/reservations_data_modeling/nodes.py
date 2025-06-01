import os
import pickle
import joblib
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
from typing import Tuple


# ──────────────────────────────────────────────
# MODEL DEFINITIONS
# ──────────────────────────────────────────────

class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x_proj = self.input_proj(x)
        x_enc = self.transformer(x_proj)
        return self.fc(x_enc[:, -1, :])


# ──────────────────────────────────────────────
# 1️ Feature engineering
# ──────────────────────────────────────────────

def prepare_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    df = df.rename(columns={'h_tot_hab': 'target_habitaciones'})
    df["day_of_week"] = df["Fecha"].dt.dayofweek
    df["month"]       = df["Fecha"].dt.month
    df["lag_1"] = df["target_habitaciones"].shift(1)
    df["lag_7"]       = df["target_habitaciones"].shift(7)
    df["lag_30"]      = df["target_habitaciones"].shift(30)
    df["rolling_mean_7"]  = df["target_habitaciones"].rolling(7).mean()
    df["rolling_trend_7"] = df["target_habitaciones"] - df["rolling_mean_7"]
    df["rolling_mean_21"] = df["target_habitaciones"].rolling(21).mean()
    df["rolling_std_30"]  = df["target_habitaciones"].rolling(30).std()
    df["day_of_year"]     = df["Fecha"].dt.dayofyear
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    mx_holidays = holidays.Mexico()
    df["is_holiday"] = df["Fecha"].isin(mx_holidays).astype(int)
    return df


# ──────────────────────────────────────────────
# 2️ Imputation & scaling
# ──────────────────────────────────────────────

def impute_and_scale(df: pd.DataFrame, features: list[str]) -> Tuple[np.ndarray, list[str]]:
    # Imputación de variables con lag o rolling usando ruido gaussiano
    for col in [c for c in features if c.startswith(("lag_", "rolling_"))]:
        np.random.seed(42)
        s = df[col].copy()
        for i in range(len(s)):
            if pd.isna(s.iat[i]):
                local = s[max(0, i - 7):i].dropna()
                s.iat[i] = (
                    np.random.normal(local.mean(), local.std())
                    if len(local) > 0 else 0
                )
        df[col] = s

    # Interpolación general
    df = df.interpolate()

    # Escalado
    scaler = StandardScaler()
    arr = scaler.fit_transform(df[features + ["target_habitaciones"]])
    os.makedirs("data/05_model_input", exist_ok=True)
    joblib.dump(scaler, "data/05_model_input/scaler.pkl")
    return arr, features, scaler

def inverse_transform_target(scaler, y_scaled: np.ndarray) -> np.ndarray:
    dummy = np.zeros((len(y_scaled), scaler.mean_.shape[0]))
    dummy[:, -1] = y_scaled
    inversed = scaler.inverse_transform(dummy)[:, -1]
    return inversed

def extract_train_test_dates(df_features: pd.DataFrame, params: dict) -> tuple[list, list]:
    periods = params["periods"]
    train_dates = df_features.iloc[:-periods]["Fecha"].tolist()
    test_dates  = df_features.iloc[-periods:]["Fecha"].tolist()
    return train_dates, test_dates


# ──────────────────────────────────────────────
# 3️ Create train/test split
# ──────────────────────────────────────────────

def create_train_test(arr: np.ndarray, params: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    class TSData(Dataset):
        def __init__(self, data, w, h):
            X, y = [], []
            for i in range(len(data) - w - h + 1):
                X.append(data[i : i + w, :-1])
                y.append(data[i + w + h - 1, -1])
            self.X = torch.tensor(np.stack(X), dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    ds = TSData(arr, params["window_size"], params["horizon"])
    splits = list(TimeSeriesSplit(n_splits=params["n_splits"]).split(ds.X))
    train_idx, test_idx = splits[-1]
    X_train, y_train = ds.X[train_idx], ds.y[train_idx]
    X_test,  y_test  = ds.X[test_idx],  ds.y[test_idx]
    return X_train, y_train, X_test, y_test


# ──────────────────────────────────────────────
# Métrica WMAPE
# ──────────────────────────────────────────────

def _compute_wmape(actual, pred):
    return np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual))


# ──────────────────────────────────────────────
# 4️ Transformer training
# ──────────────────────────────────────────────

def train_transformer(X_train, y_train, X_test, y_test, cfg: dict, scaler, seed=42, save_model_path=None) -> tuple[float, float, float]:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Determine model type (in case save_model_path is None → fallback to name)
    if save_model_path and "gru" in save_model_path:
        model_type = "gru"
    else:
        model_type = "transformer"

    best_params_path = f"data/07_optuna/{model_type}_best_params.yml"

    if os.path.exists(best_params_path):
        print(f"✅ Loading best params from {best_params_path}")
        with open(best_params_path, "r") as f:
            best_params = yaml.safe_load(f)

        cfg.update(best_params)
        cfg["epochs"] = 250
        print(f"Using BEST params + epochs={cfg['epochs']}")
    else:
        print(f"⚠️ No best params found → using cfg from parameters.yml")

    with mlflow.start_run(run_name="Transformer", nested=True):
        mlflow.log_params(cfg)

        model = TransformerForecaster(
            input_dim=cfg["input_dim"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
        ).to("cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        loss_fn = nn.MSELoss()

        for _ in range(cfg["epochs"]):
            model.train()
            for xb, yb in DataLoader(list(zip(X_train, y_train)), batch_size=cfg["batch_size"]):
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Eval
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(list(zip(X_test, y_test)), batch_size=cfg["batch_size"]):
                p = model(xb).numpy().flatten()
                preds.extend(p)
                acts.extend(yb.numpy().flatten())

        preds = np.array(preds)
        acts = np.array(acts)

        # Metrics
        preds_orig = inverse_transform_target(scaler, preds)
        acts_orig = inverse_transform_target(scaler, acts)

        mae_orig   = mean_absolute_error(acts_orig, preds_orig)
        rmse_orig  = mean_squared_error(acts_orig, preds_orig) ** 0.5
        wmape_orig = _compute_wmape(acts_orig, preds_orig)

        mlflow.log_metrics({
            "transformer_MAE_original": mae_orig,
            "transformer_RMSE_original": rmse_orig,
            "transformer_WMAPE_original": wmape_orig
        })

        # 🚀 Save model
        os.makedirs("data/06_models", exist_ok=True)
        if save_model_path:
            print(f"✅ Saving model to {save_model_path}")
            torch.save(model.state_dict(), save_model_path)
        else:
            # Default save path
            torch.save(model.state_dict(), "data/06_models/transformer_best.pt")

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "transformer_model")

        return mae_orig, rmse_orig, wmape_orig


# ──────────────────────────────────────────────
# 5️ GRU training
# ──────────────────────────────────────────────

def train_gru(X_train, y_train, X_test, y_test, cfg: dict, scaler, seed=42, save_model_path=None) -> tuple[float, float, float]:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
     # Determine model type (in case save_model_path is None → fallback to name)
    if save_model_path and "gru" in save_model_path:
        model_type = "gru"
    else:
        model_type = "transformer"

    best_params_path = f"data/07_optuna/{model_type}_best_params.yml"

    if os.path.exists(best_params_path):
        print(f"✅ Loading best params from {best_params_path}")
        with open(best_params_path, "r") as f:
            best_params = yaml.safe_load(f)

        cfg.update(best_params)
        cfg["epochs"] = 250
        print(f"Using BEST params + epochs={cfg['epochs']}  ")
    else:
        print(f"⚠️ No best params found → using cfg from parameters.yml")

    with mlflow.start_run(run_name="GRU", nested=True):
        mlflow.log_params(cfg)

        model = GRUForecaster(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
        ).to("cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        loss_fn = nn.MSELoss()

        for _ in range(cfg["epochs"]):
            model.train()
            for xb, yb in DataLoader(list(zip(X_train, y_train)), batch_size=cfg["batch_size"]):
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Eval
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(list(zip(X_test, y_test)), batch_size=cfg["batch_size"]):
                p = model(xb).numpy().flatten()
                preds.extend(p)
                acts.extend(yb.numpy().flatten())

        preds = np.array(preds)
        acts = np.array(acts)

        # Inverse transform
        preds_orig = inverse_transform_target(scaler, preds)
        acts_orig = inverse_transform_target(scaler, acts)

        mae_orig   = mean_absolute_error(acts_orig, preds_orig)
        rmse_orig  = mean_squared_error(acts_orig, preds_orig) ** 0.5
        wmape_orig = _compute_wmape(acts_orig, preds_orig)

        mlflow.log_metrics({
            "gru_MAE_original": mae_orig,
            "gru_RMSE_original": rmse_orig,
            "gru_WMAPE_original": wmape_orig
        })

        # 🚀 Save model
        os.makedirs("data/06_models", exist_ok=True)
        if save_model_path:
            print(f"✅ Saving model to {save_model_path}")
            torch.save(model.state_dict(), save_model_path)
        else:
            # Default save path
            torch.save(model.state_dict(), "data/06_models/gru_best.pt")

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "gru_model")

        return mae_orig, rmse_orig, wmape_orig



# ──────────────────────────────────────────────
# 6️ Prophet training
# ──────────────────────────────────────────────
def train_prophet(df_features: pd.DataFrame, cfg: dict) -> tuple[float, float, float]:
    import random
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)
    
    with mlflow.start_run(run_name="Prophet", nested=True):
        mlflow.log_params(cfg)

        df_p = df_features.rename(columns={"Fecha": "ds", "target_habitaciones": "y"})
        df_p["is_weekend"] = df_p["ds"].dt.weekday.isin([5, 6]).astype(int)
        mx = holidays.Mexico()
        df_p["is_holiday"] = df_p["ds"].isin(mx).astype(int)

        # 💥 Add lag & rolling features
        regressors = [
            "is_weekend",
            "is_holiday",
            "lag_7",
            "lag_30",
            "rolling_mean_7",
            "rolling_mean_21",
            "rolling_std_30",
        ]

        df_train = df_p.iloc[:-cfg["periods"]].copy()
        df_test  = df_p.iloc[-cfg["periods"]:].copy()
        df_train = df_train.dropna(subset=regressors + ["y", "ds"]).copy()
        df_test  = df_test.dropna(subset=regressors + ["y", "ds"]).copy()

        # Initialize Prophet
        m = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False
        )
        for reg in regressors:
            m.add_regressor(reg)

        # Fit model
        m.fit(df_train)

        # Prepare future dataframe
        future = df_test[["ds"] + regressors]
        forecast = m.predict(future)

        # Metrics
        y_true = df_test["y"].values
        y_pred = forecast["yhat"].values

        mae   = mean_absolute_error(y_true, y_pred)
        rmse  = mean_squared_error(y_true, y_pred) ** 0.5
        wmape = _compute_wmape(y_true, y_pred)

        # Log metrics
        mlflow.log_metrics({
            "prophet_MAE": mae,
            "prophet_RMSE": rmse,
            "prophet_WMAPE": wmape,
        })
        mlflow.log_figure(m.plot(forecast), "forecast.png")
        mlflow.log_figure(m.plot_components(forecast), "components.png")

        # Save model
        os.makedirs("data/06_models", exist_ok=True)
        prophet_path = "data/06_models/prophet_model.pkl"
        with open(prophet_path, "wb") as f:
            pickle.dump(m, f)

        # Log Prophet model
        mlflow.prophet.log_model(m, artifact_path="prophet_model")

        # 🟢 THIS is the line that was missing!
        return mae, rmse, wmape
    

def run_best_params_training(
    X_train, y_train, X_test, y_test, cfg, scaler,
    preview_epochs=10,
    final_epochs=100,
    save_preview_model_path=None,
    save_final_model_path=None,
    seed=42,
    model_type="transformer"
):
    # 1️⃣ First run: preview training
    print(f"🚀 Running PREVIEW training for {preview_epochs} epochs...")
    cfg_preview = cfg.copy()
    cfg_preview["epochs"] = preview_epochs

    if model_type == "transformer":
        train_transformer(
            X_train, y_train, X_test, y_test,
            cfg_preview, scaler, seed=seed,
            save_model_path=save_preview_model_path
        )
    elif model_type == "gru":
        train_gru(
            X_train, y_train, X_test, y_test,
            cfg_preview, scaler, seed=seed,
            save_model_path=save_preview_model_path
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 2️⃣ Final run: 100 epochs
    print(f"🚀 Running FINAL training for {final_epochs} epochs...")
    cfg_final = cfg.copy()
    cfg_final["epochs"] = final_epochs

    if model_type == "transformer":
        mae, rmse, wmape = train_transformer(
            X_train, y_train, X_test, y_test,
            cfg_final, scaler, seed=seed,
            save_model_path=save_final_model_path
        )
    elif model_type == "gru":
        mae, rmse, wmape = train_gru(
            X_train, y_train, X_test, y_test,
            cfg_final, scaler, seed=seed,
            save_model_path=save_final_model_path
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 🚀 Return final metrics so retrain_best.py can log them!
    return mae, rmse, wmape

