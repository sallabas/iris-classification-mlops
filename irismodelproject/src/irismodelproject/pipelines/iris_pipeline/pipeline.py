from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    load_data,
    split_data,
    train_model,
    evaluate_model,
    save_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs="iris_dataset",
                outputs=["X", "y"],
                name="load_data_node",
            ),
            node(
                func=split_data,
                inputs=["X", "y"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "X_test", "y_test"],
                outputs="model_accuracy",
                name="evaluate_model_node",
            ),
            node(
                func=save_model,
                inputs="trained_model",
                outputs=None,
                name="save_model_node",
            ),
        ]
    )
