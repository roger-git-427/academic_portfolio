# nodes.py
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
from prophet import Prophet
import holidays

# 1️ Feature engineering
def prepare_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    df = df.rename(columns={'h_tot_hab': 'target_habitaciones'})
    df["day_of_week"] = df["Fecha"].dt.dayofweek
    df["month"]       = df["Fecha"].dt.month
    df["lag_7"]       = df["target_habitaciones"].shift(7)
    df["lag_30"]      = df["target_habitaciones"].shift(30)
    df["rolling_mean_7"]  = df["target_habitaciones"].rolling(7).mean()
    df["rolling_mean_21"] = df["target_habitaciones"].rolling(21).mean()
    df["rolling_std_30"]  = df["target_habitaciones"].rolling(30).std()
    df["day_of_year"]     = df["Fecha"].dt.dayofyear
    df["sin_day"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_day"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df

# 2 Imputation & scaling
def impute_and_scale(
    df: pd.DataFrame,
    features: list[str]
) -> tuple[np.ndarray, list[str]]:
    # stochastic impute for each lag/rolling column
    for col in [c for c in features if c.startswith(("lag_", "rolling_"))]:
        np.random.seed(42)
        s = df[col].copy()
        for i in range(len(s)):
            if pd.isna(s.iat[i]):
                local = s[max(0, i - 7) : i].dropna()
                s.iat[i] = (
                    np.random.normal(local.mean(), local.std())
                    if len(local) > 0
                    else 0
                )
        df[col] = s
    # linear interpolate any remaining
    df = df.interpolate()

    # scale
    scaler = StandardScaler()
    arr = scaler.fit_transform(df[features + ["target_habitaciones"]])
    return arr, features

# 3️ Create train/test split
def create_train_test(
    arr: np.ndarray, params: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

def _compute_wmape(actual, pred):
    return np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual))

# 4️ Transformer training

def train_transformer(
    X_train, y_train, X_test, y_test, cfg: dict
) -> tuple[float, float, float]:
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
                loss_fn(model(xb), yb).backward()
                optimizer.step()
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(list(zip(X_test, y_test)), batch_size=cfg["batch_size"]):
                p = model(xb).numpy().flatten()
                preds.extend(p); acts.extend(yb.numpy().flatten())
        mae   = mean_absolute_error(acts, preds)
        rmse  = mean_squared_error(acts, preds, squared=False)
        wmape = _compute_wmape(np.array(acts), np.array(preds))
        mlflow.log_metrics({"transformer_MAE": mae, "transformer_RMSE": rmse, "transformer_WMAPE": wmape})
        mlflow.pytorch.log_model(model, "transformer_model")
    return mae, rmse, wmape

# 5️ GRU training
def train_gru(
    X_train, y_train, X_test, y_test, cfg: dict
) -> tuple[float, float, float]:
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
                loss_fn(model(xb), yb).backward()
                optimizer.step()
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(list(zip(X_test, y_test)), batch_size=cfg["batch_size"]):
                p = model(xb).numpy().flatten()
                preds.extend(p); acts.extend(yb.numpy().flatten())
        mae   = mean_absolute_error(acts, preds)
        rmse  = mean_squared_error(acts, preds, squared=False)
        wmape = _compute_wmape(np.array(acts), np.array(preds))
        mlflow.log_metrics({"gru_MAE": mae, "gru_RMSE": rmse, "gru_WMAPE": wmape})
        mlflow.pytorch.log_model(model, "gru_model")
    return mae, rmse, wmape

# 6️ Prophet training
def train_prophet(df_features: pd.DataFrame, cfg: dict) -> tuple[float, float, float]:
    with mlflow.start_run(run_name="Prophet", nested=True):
        mlflow.log_params(cfg)

        df_p = df_features.rename(columns={"Fecha": "ds", "target_habitaciones": "y"})
        df_p["is_weekend"] = df_p["ds"].dt.weekday.isin([5,6]).astype(int)
        mx = holidays.Mexico()
        df_p["is_holiday"] = df_p["ds"].isin(mx).astype(int)

        # Separar train y test
        df_train = df_p.iloc[:-cfg["periods"]]
        df_test  = df_p.iloc[-cfg["periods"]:]

        # Entrenar modelo
        m = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False
        )
        m.add_regressor("is_weekend")
        m.add_regressor("is_holiday")
        m.fit(df_train)

        # Hacer predicción
        future = df_test[["ds", "is_weekend", "is_holiday"]]  # ya tienen fechas reales
        forecast = m.predict(future)

        # Evaluación
        y_true = df_test["y"].values
        y_pred = forecast["yhat"].values

        mae   = mean_absolute_error(y_true, y_pred)
        rmse  = mean_squared_error(y_true, y_pred, squared=False)
        wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

        mlflow.log_metrics({
            "prophet_MAE": mae,
            "prophet_RMSE": rmse,
            "prophet_WMAPE": wmape
        })

        mlflow.log_figure(m.plot(forecast), "forecast.png")
        mlflow.log_figure(m.plot_components(forecast), "components.png")

        return mae, rmse, wmape

