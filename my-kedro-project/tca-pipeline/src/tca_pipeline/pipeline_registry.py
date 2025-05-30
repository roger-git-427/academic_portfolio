"""Project pipelines."""

from typing import Dict
from kedro.pipeline import Pipeline
from tca_pipeline.pipelines import reservations_preprocessing as rp
from tca_pipeline.pipelines import reservations_data_modeling as rdm
from tca_pipeline.pipelines import reservations_inference as ri



def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "__default__": rp.create_pipeline(),
        "reservations_preprocessing": rp.create_pipeline(),
        "reservations_data_modeling": rdm.create_pipeline(),
        "reservations_inference": ri.create_pipeline(),
    }
