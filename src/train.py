# src/train.py

from src.data_loader import load_and_save_iris_dataset
from src.model import build_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():
    """Train the Iris classification model and save it."""
    df = load_and_save_iris_dataset()
    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Model accuracy: {acc:.2f}")

    save_model(model)
    print("🏁 Training complete.")


if __name__ == "__main__":
    train_model()
