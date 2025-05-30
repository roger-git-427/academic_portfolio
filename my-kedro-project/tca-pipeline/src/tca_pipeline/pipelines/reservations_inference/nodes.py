import pandas as pd
import torch

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
from tca_pipeline.pipelines.reservations_data_modeling.nodes import (
    prepare_features,
    impute_and_scale,
)

def load_model() -> torch.nn.Module:
    """Load the Transformer model logged in MLflow."""
    # uses the catalog entry `transformer_model` below
    return models.MlflowModelTrackingDataset().load()

def inference_features(rooms_by_date: pd.DataFrame, scaler, features: list[str]) -> torch.Tensor:
    """Re-use your training featurization & scaling."""
    df_feat = prepare_features(rooms_by_date)
    arr, _ = impute_and_scale(df_feat, features)
    return torch.tensor(arr, dtype=torch.float32)

def predict_transformer(features: torch.Tensor, model: torch.nn.Module) -> pd.DataFrame:
    """Run the model and return a simple DataFrame with Fecha & prediction."""
    model.eval()
    with torch.no_grad():
        # assume model expects shape [batch, seq, feat]; adjust if different
        X = features.unsqueeze(0)
        preds = model(X).cpu().numpy().flatten()
    return pd.DataFrame({"prediction": preds})
