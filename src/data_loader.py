from sklearn.datasets import load_iris
import pandas as pd
import os


def load_and_save_iris_dataset():
    """
    Loads the Iris dataset and saves it into the main data/raw.
    """

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(project_root, "data", "raw", "iris.csv")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)
    print(f"Dataset saved successfully at: {save_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Current working directory: {os.getcwd()}")

    return df


if __name__ == "__main__":
    load_and_save_iris_dataset()
