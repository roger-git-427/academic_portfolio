import pandas as pd
import numpy as np
import torch
from typing import List
from sklearn.preprocessing import StandardScaler

from tca_pipeline.pipelines.reservations_data_modeling.nodes import prepare_features

from tca_pipeline.pipelines.reservations_preprocessing.nodes import (
    select_columns,
    merge_lookup_tables,
    convert_dates,
    enforce_types_and_basic_filters,
    normalise_city,
    remove_outliers_percentile,
    replace_h_num_persons,
    filtered_df,
    build_daily_occupancy,
)


def inference_features(
    rooms_by_date: pd.DataFrame,
    scaler: StandardScaler,
    features: List[str],
) -> torch.Tensor:
    """
    1) Prepare features exactly as in training.
    2) Impute missing lag_*/rolling_* columns with the same logic.
    3) Interpolate remaining NaNs.
    4) APPLY the pre-fitted scaler (no .fit!).
    5) Drop the target column and return a torch.Tensor of shape (seq_len, n_features).
    """
    # — Step 1: Feature engineering —
    print(f"[DEBUG] Step 1 - raw input shape: {rooms_by_date.shape}")
    df = prepare_features(rooms_by_date)
    print(f"[DEBUG] Step 2 - after prepare_features: {df.shape}")

    # — Step 2: Impute lag_/rolling_ columns —
    for col in [c for c in features if c.startswith(("lag_", "rolling_"))]:
        np.random.seed(42)
        s = df[col].copy()
        for i in range(len(s)):
            if pd.isna(s.iat[i]):
                window = s[max(0, i - 7):i].dropna()
                s.iat[i] = np.random.normal(window.mean(), window.std()) if len(window) else 0
        df[col] = s
    print(f"[DEBUG] Step 3 - after imputation: {df[features].shape}")

    # — Step 3: Interpolation —
    df = df.interpolate()
    print(f"[DEBUG] Step 4 - after interpolation: {df[features].shape}")

    # — Step 4: Scaling —
    arr = scaler.transform(df[features + ["target_habitaciones"]])
    print(f"[DEBUG] Step 5 - after scaling (with target): {arr.shape}")

    # — Step 5: Drop the target column —
    X_arr = arr[:, : len(features)]
    print(f"[DEBUG] Step 6 - final input to model (no target): {X_arr.shape}")

    return torch.tensor(X_arr, dtype=torch.float32)


def predict_transformer(
    features: torch.Tensor,
    model: torch.nn.Module
) -> pd.DataFrame:
    """
    features must be shape (seq_len, n_features).
    We add a batch dimension (1, seq_len, n_features) to run through the model.
    """

    # 1) Print how many features the loaded model expects:
    try:
        expected_in = model.input_proj.in_features
    except AttributeError:
        expected_in = None
    print(f">>> Loaded Transformer expects input_proj.in_features = {expected_in}")

    # 2) Print the actual feature‐tensor shape we’re passing in:
    print(f">>> features_inf tensor shape: {features.shape}")

    # 3) Unsqueeze and forward‐pass:
    model.eval()
    with torch.no_grad():
        X = features.unsqueeze(0)
        print(f"[DEBUG] Model input shape (after unsqueeze): {X.shape}")
        preds = model(X).cpu().numpy().flatten()

    print(f"[DEBUG] Predictions shape: {preds.shape}")
    return pd.DataFrame({"prediction": preds})
