from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data, split_and_scale_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs=["wine_raw", "params:target_threshold"],
            outputs="wine_cleaned",
            name="clean_data_node",
        ),
        node(
            func=split_and_scale_data,
            inputs=["wine_cleaned", "parameters"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_and_scale_data_node",
        ),
    ])
