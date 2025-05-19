"""
This is a boilerplate pipeline 'reservations_preprocessing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import (
    select_columns,
    merge_lookup_tables,
    convert_dates,
    enforce_types_and_basic_filters,
    normalise_city,
    remove_outliers_percentile,
    explode_and_sum_rooms,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: D401
    return pipeline([
        node(select_columns,        "raw_reservaciones",            "reservations_base",          name="select_cols"),
        node(merge_lookup_tables,      dict(
                                            reservations="reservations_base",
                                            canales="raw_canales",
                                            empresas="raw_empresas",
                                            agencias="raw_agencias",
                                            estatus="raw_estatus_reservaciones"
                                        ),
                                        "reservations_merged",         name="merge_lookups"),
        node(convert_dates,            "reservations_merged",           "reservations_dates",         name="convert_dates"),
        node(enforce_types_and_basic_filters,
                                        "reservations_dates",           "reservations_typed",         name="types_filters"),
        node(normalise_city,           "reservations_typed",          "reservations_city_norm",     name="normalise_city"),
        node(
            remove_outliers_percentile,
            inputs=[
                "reservations_city_norm",
                "params:outlier_exclude_cols",
                "params:outlier_pct"
            ],
            outputs="reservations_clean",
            name="outliers_pct"
        ),
        node(explode_and_sum_rooms,    "reservations_clean",            "rooms_by_date",             name="explode_sum"),
    ])
