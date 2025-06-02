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
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
# from neuralforecast.core import NeuralForecast
# from neuralforecast.models import TimesNet
# from neuralforecast.losses.pytorch import MAE, MSE


# MODEL DEFINITIONS
# 


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
        plt.savefig(save_path)
    plt.close()


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

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])



# 2️ Imputation & scaling 

def impute_and_scale(df: pd.DataFrame, features: list[str]) -> Tuple[np.ndarray, list[str], StandardScaler]:
    """
    1) Impute missing lag_/rolling_ columns using Gaussian noise from last 7 observed
    2) Interpolate any remaining NaNs
    3) Scale all 'features' + 'target_habitaciones' using StandardScaler
    4) Save the scaler to disk (pickle)
    """
    print(f"[DEBUG][impute_and_scale] Input df shape: {df.shape}")
    print(f"[DEBUG][impute_and_scale] Features list: {features}")

    # Imputation of lag_*/rolling_ using Gaussian noise
    for col in [c for c in features if c.startswith(("lag_", "rolling_"))]:
        np.random.seed(42)
        s = df[col].copy()
        for i in range(len(s)):
            if pd.isna(s.iat[i]):
                window = s[max(0, i - 7):i].dropna()
                s.iat[i] = np.random.normal(window.mean(), window.std()) if len(window) else 0
        df[col] = s
    print(f"[DEBUG][impute_and_scale] After imputation, subset df[features] shape: {df[features].shape}")

    # Interpolation for any remaining NaNs
    df = df.interpolate()
    print(f"[DEBUG][impute_and_scale] After interpolation, subset df[features] shape: {df[features].shape}")

    # Scaling
    scaler = StandardScaler()
    arr = scaler.fit_transform(df[features + ["target_habitaciones"]])
    print(f"[DEBUG][impute_and_scale] Scaled array shape: {arr.shape}")

    # Save the scaler for inference later
    os.makedirs("data/05_model_input", exist_ok=True)
    scaler_path = "data/05_model_input/scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[DEBUG][impute_and_scale] Saved scaler (pickle) to: {scaler_path}")

    return arr, features, scaler


def inverse_transform_target(scaler: StandardScaler, y_scaled: np.ndarray) -> np.ndarray:
    """
    Given a 1D array y_scaled, reconstruct an array of zeros except y_scaled at last column
    to call scaler.inverse_transform(dummy) and extract the last column back to original scale.
    """
    dummy = np.zeros((len(y_scaled), scaler.mean_.shape[0]))
    dummy[:, -1] = y_scaled
    inversed = scaler.inverse_transform(dummy)[:, -1]
    return inversed


def extract_train_test_dates(df_features: pd.DataFrame, params: dict) -> tuple[list, list]:
    """
    Return two lists: train_dates = all Date values except last 'periods';
                     test_dates = last 'periods' Date values.
    """
    periods = params["periods"]
    train_dates = df_features.iloc[:-periods]["Fecha"].tolist()
    test_dates  = df_features.iloc[-periods:]["Fecha"].tolist()
    return train_dates, test_dates


# 
# 3️ Create train/test split (with debug prints)
# 

def create_train_test(arr: np.ndarray, params: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    1) Turn `arr` into a sequence dataset of window_size/horizon
    2) Perform a TimeSeriesSplit, take the last split for train/test
    3) Return: X_train, y_train, X_test, y_test as torch.Tensors
    """
    print(f"[DEBUG][create_train_test] Input arr shape: {arr.shape}")
    window_size = params["window_size"]
    horizon = params["horizon"]
    n_splits = params["n_splits"]
    print(f"[DEBUG][create_train_test] window_size={window_size}, horizon={horizon}, n_splits={n_splits}")

    class TSData(Dataset):
        def __init__(self, data, w, h):
            X, y = [], []
            for i in range(len(data) - w - h + 1):
                X.append(data[i: i + w, :-1])
                y.append(data[i + w + h - 1, -1])
            self.X = torch.tensor(np.stack(X), dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    ds = TSData(arr, window_size, horizon)
    print(f"[DEBUG][create_train_test] Dataset size: {len(ds)}, each X shape: {ds.X.shape}, each y shape: {ds.y.shape}")

    splits = list(TimeSeriesSplit(n_splits=n_splits).split(ds.X))
    train_idx, test_idx = splits[-1]
    X_train, y_train = ds.X[train_idx], ds.y[train_idx]
    X_test,  y_test  = ds.X[test_idx],  ds.y[test_idx]
    print(f"[DEBUG][create_train_test] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[DEBUG][create_train_test] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test


# 
# 4️ Transformer training (with debug prints + pickle saving)
# 

def _compute_wmape(actual, pred):
    return np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual))


def train_transformer(
    X_train, y_train, X_test, y_test, cfg: dict, scaler, seed=42, save_model_path=None
) -> tuple[float, float, float]:
    """
    1) Load best Optuna params if they exist
    2) Train a TransformerForecaster
    3) Log metrics to MLflow, save the final model with pickle
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_type = "transformer"
    best_params_path = f"data/07_optuna/{model_type}_best_params.yml"

    # — Load best parameters if available —
    if os.path.exists(best_params_path):
        print(f"[DEBUG][train_transformer] Loading best params from {best_params_path}")
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

    with mlflow.start_run(run_name="Transformer", nested=True):
        mlflow.log_params(cfg)

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

        # — Training loop —
        for epoch in range(cfg["epochs"]):
            model.train()
            for xb, yb in DataLoader(list(zip(X_train, y_train)), batch_size=cfg["batch_size"]):
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
            if epoch % 50 == 0 or epoch == cfg["epochs"] - 1:
                print(f"[DEBUG][train_transformer] Epoch {epoch+1}/{cfg['epochs']}: loss={loss.item():.4f}")

        # — Evaluation —
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(list(zip(X_test, y_test)), batch_size=cfg["batch_size"]):
                p = model(xb).numpy().flatten()
                preds.extend(p)
                acts.extend(yb.numpy().flatten())

        preds = np.array(preds)
        acts = np.array(acts)

        # — Inverse transform to original scale —
        preds_orig = inverse_transform_target(scaler, preds)
        acts_orig = inverse_transform_target(scaler, acts)

        mae_orig = mean_absolute_error(acts_orig, preds_orig)
        rmse_orig = mean_squared_error(acts_orig, preds_orig) ** 0.5
        wmape_orig = _compute_wmape(acts_orig, preds_orig)

        print(f"[DEBUG][train_transformer] Evaluation MAE={mae_orig:.4f}, RMSE={rmse_orig:.4f}, WMAPE={wmape_orig:.4f}")

        mlflow.log_metrics({
            "transformer_MAE_original": mae_orig,
            "transformer_RMSE_original": rmse_orig,
            "transformer_WMAPE_original": wmape_orig
        })

        # — Save the entire model object via pickle —
        os.makedirs("data/06_models", exist_ok=True)
        final_path = save_model_path or "data/06_models/transformer_model.pkl"
        with open(final_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[DEBUG][train_transformer] Saved Transformer model (pickle) to: {final_path}")

        # — Log model artifact to MLflow —
        mlflow.pytorch.log_model(model, "transformer_model")

        # Plot
        plot_path = "data/06_models/transformer_forecast.png"
        plot_forecast_vs_actual(acts_orig, preds_orig, title="Transformer Forecast vs Actual", save_path=plot_path)
        mlflow.log_artifact(plot_path)

        return mae_orig, rmse_orig, wmape_orig



# 
# 5️ GRU training (with debug prints + pickle saving)
# 

def train_gru(
    X_train, y_train, X_test, y_test, cfg: dict, scaler, seed=42, save_model_path=None
) -> tuple[float, float, float]:
    """
    1) Load best Optuna params if they exist
    2) Train a GRUForecaster
    3) Log metrics to MLflow, save the final model with pickle
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    model_type = "gru"
    best_params_path = f"data/07_optuna/{model_type}_best_params.yml"

    if os.path.exists(best_params_path):
        print(f"[DEBUG][train_gru] Loading best params from {best_params_path}")
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

    with mlflow.start_run(run_name="GRU", nested=True):
        mlflow.log_params(cfg)

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

        for epoch in range(cfg["epochs"]):
            model.train()
            for xb, yb in DataLoader(list(zip(X_train, y_train)), batch_size=cfg["batch_size"]):
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
            if epoch % 50 == 0 or epoch == cfg["epochs"] - 1:
                print(f"[DEBUG][train_gru] Epoch {epoch+1}/{cfg['epochs']}: loss={loss.item():.4f}")

        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(list(zip(X_test, y_test)), batch_size=cfg["batch_size"]):
                p = model(xb).numpy().flatten()
                preds.extend(p)
                acts.extend(yb.numpy().flatten())

        preds = np.array(preds)
        acts = np.array(acts)

        preds_orig = inverse_transform_target(scaler, preds)
        acts_orig = inverse_transform_target(scaler, acts)

        mae_orig = mean_absolute_error(acts_orig, preds_orig)
        rmse_orig = mean_squared_error(acts_orig, preds_orig) ** 0.5
        wmape_orig = _compute_wmape(acts_orig, preds_orig)

        print(f"[DEBUG][train_gru] Evaluation MAE={mae_orig:.4f}, RMSE={rmse_orig:.4f}, WMAPE={wmape_orig:.4f}")

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

        # Plot
        plot_path = "data/06_models/gru_forecast.png"
        plot_forecast_vs_actual(acts_orig, preds_orig, title="GRU Forecast vs Actual", save_path=plot_path)
        mlflow.log_artifact(plot_path)

        return mae_orig, rmse_orig, wmape_orig

def train_lstm(X_train, y_train, X_test, y_test, cfg: dict, scaler, seed=42, save_model_path=None) -> tuple[float, float, float]:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_type = "lstm"
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

    cfg["input_dim"] = X_train.shape[-1]
    print(f"👉 LSTM input_dim = {cfg['input_dim']}")

    with mlflow.start_run(run_name="LSTM", nested=True):
        mlflow.log_params(cfg)

        model = LSTMForecaster(
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

        preds_orig = inverse_transform_target(scaler, preds)
        acts_orig = inverse_transform_target(scaler, acts)

        mae_orig = mean_absolute_error(acts_orig, preds_orig)
        rmse_orig = mean_squared_error(acts_orig, preds_orig) ** 0.5
        wmape_orig = _compute_wmape(acts_orig, preds_orig)

        mlflow.log_metrics({
            "lstm_MAE_original": mae_orig,
            "lstm_RMSE_original": rmse_orig,
            "lstm_WMAPE_original": wmape_orig
        })

        os.makedirs("data/06_models", exist_ok=True)
        if save_model_path:
            print(f"✅ Saving model to {save_model_path}")
            torch.save(model.state_dict(), save_model_path)
            pickle_path = save_model_path.replace(".pt", ".pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(model, f)
        else:
            default_pt_path = "data/06_models/lstm_best.pt"
            default_pkl_path = "data/06_models/lstm_best.pkl"
            torch.save(model.state_dict(), default_pt_path)
            with open(default_pkl_path, "wb") as f:
                pickle.dump(model, f)

        mlflow.pytorch.log_model(model, "lstm_model")

        # Plot
        plot_path = "data/06_models/lstm_forecast.png"
        plot_forecast_vs_actual(acts_orig, preds_orig, title="LSTM Forecast vs Actual", save_path=plot_path)
        mlflow.log_artifact(plot_path)

        return mae_orig, rmse_orig, wmape_orig


# 
# 6️ Prophet training (with debug prints)
# 

def train_prophet(df_features: pd.DataFrame, cfg: dict) -> tuple[float, float, float]:
    """
    1) Train a Prophet model (with external regressors)
    2) Log metrics to MLflow, save the final model via pickle
    """
    import random
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    print(f"[DEBUG][train_prophet] Input df_features shape: {df_features.shape}")
    print(f"[DEBUG][train_prophet] Prophet config (periods): {cfg['periods']}")

    with mlflow.start_run(run_name="Prophet", nested=True):
        mlflow.log_params(cfg)

        # Prepare data for Prophet
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
        print(f"[DEBUG][train_prophet] Regressors used: {regressors}")

        df_train = df_p.iloc[:-cfg["periods"]].copy()
        df_test  = df_p.iloc[-cfg["periods"]:].copy()
        df_train = df_train.dropna(subset=regressors + ["y", "ds"]).copy()
        df_test  = df_test.dropna(subset=regressors + ["y", "ds"]).copy()

        print(f"[DEBUG][train_prophet] df_train shape (after dropna): {df_train.shape}")
        print(f"[DEBUG][train_prophet] df_test shape (after dropna): {df_test.shape}")

        # Initialize and add regressors
        m = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )

        for reg in regressors:
            m.add_regressor(reg)

        # Fit model
        m.fit(df_train)
        print(f"[DEBUG][train_prophet] Completed Prophet.fit()")

        # Prepare future DataFrame and predict
        future = df_test[["ds"] + regressors]
        forecast = m.predict(future)
        print(f"[DEBUG][train_prophet] Completed Prophet.predict(), forecast shape: {forecast.shape}")

        # Compute metrics
        y_true = df_test["y"].values
        y_pred = forecast["yhat"].values

        mae   = mean_absolute_error(y_true, y_pred)
        rmse  = mean_squared_error(y_true, y_pred) ** 0.5
        wmape = _compute_wmape(y_true, y_pred)

        print(f"[DEBUG][train_prophet] Evaluation MAE={mae:.4f}, RMSE={rmse:.4f}, WMAPE={wmape:.4f}")

        mlflow.log_metrics({
            "prophet_MAE": mae,
            "prophet_RMSE": rmse,
            "prophet_WMAPE": wmape,
        })

        # Log figures to MLflow

        # Custom forecast plot — consistent with GRU / LSTM / SARIMA
        plot_path = "data/06_models/prophet_forecast.png"
        plot_forecast_vs_actual(y_true, y_pred, title="Prophet Forecast vs Actual", save_path=plot_path)
        mlflow.log_artifact(plot_path)

        # Keep components plot if you want
        mlflow.log_figure(m.plot_components(forecast), "prophet_components.png")


        # Save Prophet model via pickle
        os.makedirs("data/06_models", exist_ok=True)
        prophet_path = "data/06_models/prophet_model.pkl"
        with open(prophet_path, "wb") as f:
            pickle.dump(m, f)
        print(f"[DEBUG][train_prophet] Saved Prophet model (pickle) to: {prophet_path}")

        mlflow.prophet.log_model(m, artifact_path="prophet_model")

        return mae, rmse, wmape
# 
# 7️ SARIMA training
# 
def train_sarima(df_features: pd.DataFrame, cfg: dict) -> tuple[float, float, float]:

    # 👉 Prepare series
    df_p = df_features.rename(columns={"Fecha": "ds", "target_habitaciones": "y"})
    df_p = df_p.set_index("ds").asfreq("D")  # Ensure daily frequency
    df_p = df_p.fillna(method="ffill")  # Fill missing with forward fill

    series = df_p["y"]

    # 👉 Split train/test
    train = series.iloc[:-cfg["periods"]]
    test  = series.iloc[-cfg["periods"]:]

    # 👉 Build SARIMA model
    print(f"✅ Training SARIMA with order={cfg['order']} and seasonal_order={cfg['seasonal_order']}")
    model = SARIMAX(
        train,
        order=tuple(cfg["order"]),
        seasonal_order=tuple(cfg["seasonal_order"]),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_result = model.fit(disp=False)

    # yForecast
    forecast = sarima_result.forecast(steps=cfg["periods"])

    #  Metrics
    y_true = test.values
    y_pred = forecast.values

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = mean_squared_error(y_true, y_pred) ** 0.5
    wmape = _compute_wmape(y_true, y_pred)

    # Log to MLflow
    with mlflow.start_run(run_name="SARIMA", nested=True):
        mlflow.log_params({
            "order": cfg["order"],
            "seasonal_order": cfg["seasonal_order"],
            "periods": cfg["periods"]
        })
        mlflow.log_metrics({
            "sarima_MAE": mae,
            "sarima_RMSE": rmse,
            "sarima_WMAPE": wmape
        })

        # Save SARIMA model
        os.makedirs("data/06_models", exist_ok=True)
        sarima_path = "data/06_models/sarima_model.pkl"
        with open(sarima_path, "wb") as f:
            pickle.dump(sarima_result, f)
        print(f"✅ Saved SARIMA model to {sarima_path}")

    print(f"✅ SARIMA → MAE: {mae:.2f}, RMSE: {rmse:.2f}, WMAPE: {wmape:.4f}")
    plot_path = "data/06_models/sarima_forecast.png"
    plot_forecast_vs_actual(y_true, y_pred, title="SARIMA Forecast vs Actual", save_path=plot_path)
    mlflow.log_artifact(plot_path)


    return mae, rmse, wmape
 


# 
# 7️ Optional helper for running best‐params training (unchanged)
# 

def run_best_params_training(
    X_train, y_train, X_test, y_test, cfg, scaler,
    preview_epochs=10,
    final_epochs=250,
    save_preview_model_path=None,
    save_final_model_path=None,
    seed=42,
    model_type="transformger"
):
    print(f"[DEBUG][run_best_params_training] Starting preview run for {model_type}")
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

    print(f"[DEBUG][run_best_params_training] Starting final run for {model_type}")
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

    return mae, rmse, wmape
