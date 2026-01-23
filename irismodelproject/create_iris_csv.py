from sklearn.datasets import load_iris
import pandas as pd
from pathlib import Path

iris = load_iris(as_frame=True)
df = iris.frame

df["species"] = df["target"]
df.drop(columns=["target"], inplace=True)

output_path = Path("data/01_raw")
output_path.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path / "iris.csv", index=False)

print("iris.csv created successfully")
