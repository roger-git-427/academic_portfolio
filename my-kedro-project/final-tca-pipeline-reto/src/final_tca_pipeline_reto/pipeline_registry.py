"""Project pipelines."""

from kedro.pipeline import Pipeline
from final_tca_pipeline_reto.pipelines import reservations_data_preprocessing as rdp


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines =  { "__default__": rdp.create_pipeline(),
        "reservations_preprocessing": rdp.create_pipeline(),}
    return pipelines
