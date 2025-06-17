# src/<your_package>/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import (
    impute_and_scale,
    create_train_test,
    train_sarima,
    train_transformer,
    train_gru,
    train_lstm,
    train_prophet
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
            node(
                func=impute_and_scale,
                inputs=["df_features", "params:modeling.features"],
                outputs=["arr", "features_list", "scaler"],
                name="impute_and_scale",
            ),
            node(
                func=create_train_test,
                inputs=["arr", "params:modeling"],
                outputs=["X_train", "y_train", "X_test", "y_test"],
                name="create_train_test",
            ),
            node(
                func=train_sarima,
                inputs=["df_features", "params:modeling.sarima", "scaler"],
                outputs=["sarima_mae", "sarima_rmse", "sarima_wmape", "sarima_model_pickle"],
                name="train_sarima",
            ),
            node(
                func=train_transformer,
                inputs=["X_train", "y_train", "X_test", "y_test", "params:modeling.transformer", "scaler"],
                outputs=["transformer_mae", "transformer_rmse", "transformer_wmape", "transformer_model", "transformer_model_pickle"],
                name="train_transformer",
            ),
            node(
                func=train_gru,
                inputs=["X_train", "y_train", "X_test", "y_test", "params:modeling.gru", "scaler"],
                outputs=["gru_mae", "gru_rmse", "gru_wmape", "gru_model", "gru_model_pickle"],
                name="train_gru",
            ),
            node(
                func=train_lstm,
                inputs=["X_train", "y_train", "X_test", "y_test", "params:modeling.lstm", "scaler"],
                outputs=["lstm_mae", "lstm_rmse", "lstm_wmape", "lstm_model", "lstm_model_pickle"],
                name="train_lstm",
            ),
            node(
                func=train_prophet,
                inputs=["df_features", "params:modeling.prophet", "scaler"],
                outputs=["prophet_mae", "prophet_rmse", "prophet_wmape", "prophet_model", "prophet_model_pickle"],
                name="train_prophet",
            ),
        ]
    )
