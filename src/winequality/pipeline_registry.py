"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    from winequality.pipelines.data_preprocessing import create_pipeline as data_preprocessing_pipeline
    from winequality.pipelines.data_science import create_pipeline as data_science_pipeline
    from winequality.pipelines.evaluation import create_pipeline as evaluation_pipeline

    pipelines = {
        "data_preprocessing": data_preprocessing_pipeline(),
        "data_science": data_science_pipeline(),
        "evaluation": evaluation_pipeline(),
    }

    # Domyślny rurociąg: suma wszystkich zarejestrowanych rurociągów
    pipelines["__default__"] = sum(pipelines.values()) if pipelines else Pipeline([])

    return pipelines
