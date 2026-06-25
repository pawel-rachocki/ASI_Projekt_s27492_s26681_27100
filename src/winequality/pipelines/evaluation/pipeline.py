from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_model,
            inputs=["baseline_logreg", "X_test", "y_test"],
            outputs="metrics",
            name="evaluate_model_node",
        ),
    ])
