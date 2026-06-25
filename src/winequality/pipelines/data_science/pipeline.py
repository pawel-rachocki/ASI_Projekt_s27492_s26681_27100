from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_baseline_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_baseline_model,
            inputs=["X_train", "y_train"],
            outputs="baseline_logreg",
            name="train_baseline_model_node",
        ),
    ])
