from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_features,
    impute_and_scale,
    create_train_test,
    train_transformer,
    train_gru,
    train_prophet
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            prepare_features,
            inputs="rooms_by_date",
            outputs="df_features",
            name="prepare_features"
        ),
        node(
            impute_and_scale,
            inputs=["df_features", "params:modeling.features"],
            outputs=["arr", "features_list", "scaler"],
            name="impute_and_scale"
        ),
        node(
            create_train_test,
            inputs=["arr", "params:modeling"],
            outputs=["X_train", "y_train", "X_test", "y_test"],
            name="create_train_test"
        ),
        node(
            train_transformer,
            inputs=["X_train", "y_train", "X_test", "y_test", "params:modeling.transformer", "scaler"],
            outputs=["transformer_mae", "transformer_rmse", "transformer_wmape"],
            name="train_transformer"
        ),
        node(
            train_gru,
            inputs=["X_train", "y_train", "X_test", "y_test", "params:modeling.gru", "scaler"],
            outputs=["gru_mae", "gru_rmse", "gru_wmape"],
            name="train_gru"
        ),
        node(
            train_prophet,
            inputs=["df_features", "params:modeling.prophet"],
            outputs=["prophet_mae", "prophet_rmse", "prophet_wmape"],
            name="train_prophet"
        )
    ])
