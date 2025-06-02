from kedro.pipeline import pipeline, node
from . import nodes


def create_pipeline(**kwargs):
    return pipeline([
        # --- Preprocessing nodes ---
        node(
            func=nodes.select_columns,
            inputs="raw_reservaciones_inference",
            outputs="reservations_base_inf",
            name="select_columns_inf"
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
            name="merge_lookups_inf"
        ),
        node(
            func=nodes.convert_dates,
            inputs="reservations_merged_inf",
            outputs="reservations_dates_inf",
            name="convert_dates_inf"
        ),
        node(
            func=nodes.enforce_types_and_basic_filters,
            inputs="reservations_dates_inf",
            outputs="reservations_typed_inf",
            name="types_and_filters_inf"
        ),
        node(
            func=nodes.normalise_city,
            inputs="reservations_typed_inf",
            outputs="reservations_city_norm_inf",
            name="normalise_city_inf"
        ),
        node(
            func=nodes.replace_h_num_persons,
            inputs="reservations_city_norm_inf",
            outputs="reservations_grouped_inf",
            name="replace_persons_inf"
        ),
        node(
            func=nodes.filtered_df,
            inputs="reservations_grouped_inf",
            outputs="reservations_filtered_inf",
            name="filter_reservations_inf"
        ),
        node(
            func=nodes.build_daily_occupancy,
            inputs=[
                "reservations_filtered_inf",
                "params:inference.START_DATE",
                "params:inference.END_DATE"
            ],
            outputs="rooms_by_date_inf",
            name="build_occupancy_inf"
        ),
        # --- Inference nodes ---
        node(
            func=nodes.inference_features,
            inputs=[
                "rooms_by_date_inf",
                "scaler",
                "params:modeling.features"
            ],
            outputs="features_inf",
            name="make_features_inf"
        ),
        node(
            func=nodes.predict_transformer,
            inputs=["features_inf", "transformer_model_pickle"],
            outputs="predictions_inf",
            name="predict_transformer_inf"
        ),
    ])
