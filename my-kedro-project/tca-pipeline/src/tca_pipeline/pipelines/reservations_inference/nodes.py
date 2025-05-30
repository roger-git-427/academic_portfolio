
import pandas as pd
import numpy as np
import torch
from typing import List
from sklearn.preprocessing import StandardScaler

from tca_pipeline.pipelines.reservations_data_modeling.nodes import prepare_features


from kedro_mlflow.io import models
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

def load_transformer_model() -> torch.nn.Module:
    """Load the PyTorch transformer model logged in MLflow."""
    from kedro_mlflow.io import models
    # This uses your catalog entry `transformer_model`
    return models.MlflowModelTrackingDataset().load()


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
    5) Return a torch.Tensor for prediction.
    """
    # 1) Feature engineering
    df = prepare_features(rooms_by_date)

    # 2) Imputation for lag_*/rolling_
    for col in [c for c in features if c.startswith(("lag_", "rolling_"))]:
        np.random.seed(42)
        s = df[col].copy()
        for i in range(len(s)):
            if pd.isna(s.iat[i]):
                window = s[max(0, i - 7) : i].dropna()
                s.iat[i] = (
                    np.random.normal(window.mean(), window.std())
                    if len(window) > 0
                    else 0
                )
        df[col] = s

    # 3) Interpolation
    df = df.interpolate()

    # 4) Scaling
    arr = scaler.transform(df[features + ["target_habitaciones"]])

    # 5) To tensor
    return torch.tensor(arr, dtype=torch.float32)


def predict_transformer(
    features: torch.Tensor, model: torch.nn.Module
) -> pd.DataFrame:
    """Run the model and return predictions as a DataFrame."""
    model.eval()
    with torch.no_grad():
        X = features.unsqueeze(0)
        preds = model(X).cpu().numpy().flatten()
    return pd.DataFrame({"prediction": preds})
