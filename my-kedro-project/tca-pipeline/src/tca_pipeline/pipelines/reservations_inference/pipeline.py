from kedro.pipeline import Pipeline, node
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        # 1) run preprocessing exactly as in your preprocessing pipeline:
        nodes=[
            node(nodes.select_columns,    "raw_reservaciones_inference", "reservations_base_inf"),
            node(nodes.merge_lookup_tables, dict(
                reservations="reservations_base_inf",
                canales="raw_canales",
                empresas="raw_empresas",
                agencias="raw_agencias",
                estatus="raw_estatus_reservaciones"
            ), "reservations_merged_inf"),
            node(nodes.convert_dates,     "reservations_merged_inf", "reservations_dates_inf"),
            node(nodes.enforce_types_and_basic_filters, "reservations_dates_inf", "reservations_typed_inf"),
            node(nodes.normalise_city,     "reservations_typed_inf", "reservations_city_norm_inf"),
            node(nodes.remove_outliers_percentile,
                 ["reservations_city_norm_inf","params:outlier_exclude_cols","params:outlier_pct"],
                 "reservations_iqr_inf"),
            node(nodes.replace_h_num_persons, "reservations_iqr_inf", "reservations_grouped_inf"),
            node(nodes.filtered_df,        "reservations_grouped_inf", "reservations_filtered_inf"),
            node(nodes.build_daily_occupancy, "reservations_filtered_inf", "rooms_by_date_inf"),
            # 2) load scaler & model from MLflow
            node(lambda: None, None, "dummy_for_order", name="align"),  # no-op to keep catalog happy
            node(nodes.load_model,         None, "transformer_model"),
            node(nodes.inference_features, "rooms_by_date_inf", "features_inf"),
            node(nodes.predict_transformer, ["features_inf","transformer_model"], "predictions_inf"),
        ]
    )
