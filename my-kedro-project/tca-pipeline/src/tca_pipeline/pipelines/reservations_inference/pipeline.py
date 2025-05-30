from kedro.pipeline import Pipeline, node
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # --- Preprocessing (reuse your existing nodes)
            node(
                func=nodes.select_columns,
                inputs="raw_reservaciones_inference",
                outputs="reservations_base_inf",
            ),
            node(
                func=nodes.merge_lookup_tables,
                inputs=dict(
                    reservations="reservations_base_inf",
                    canales="raw_canales",
                    empresas="raw_empresas",
                    agencias="raw_agencias",
                    estatus="raw_estatus_reservaciones",
                ),
                outputs="reservations_merged_inf",
            ),
            node(
                func=nodes.convert_dates,
                inputs="reservations_merged_inf",
                outputs="reservations_dates_inf",
            ),
            node(
                func=nodes.enforce_types_and_basic_filters,
                inputs="reservations_dates_inf",
                outputs="reservations_typed_inf",
            ),
            node(
                func=nodes.normalise_city,
                inputs="reservations_typed_inf",
                outputs="reservations_city_norm_inf",
            ),
            node(
                func=nodes.replace_h_num_persons,
                inputs="reservations_city_norm_inf",
                outputs="reservations_grouped_inf",
            ),
            node(
                func=nodes.filtered_df,
                inputs="reservations_grouped_inf",
                outputs="reservations_filtered_inf",
            ),
            node(
                func=nodes.build_daily_occupancy,
                inputs="reservations_filtered_inf",
                outputs="rooms_by_date_inf",
            ),

            # --- Inference: load scaler & model, then predict
            node(
                func=nodes.load_transformer_model,
                inputs=None,
                outputs="transformer_model",
                name="load_model",
            ),
            node(
                func=nodes.inference_features,
                inputs=["rooms_by_date_inf", "scaler", "params:features"],
                outputs="features_inf",
                name="make_features",
            ),
            node(
                func=nodes.predict_transformer,
                inputs=["features_inf", "transformer_model"],
                outputs="predictions_inf",
                name="predict",
            ),
        ]
    )
