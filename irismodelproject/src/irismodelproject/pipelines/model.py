# src/model.py

from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def build_model(random_state=10):
    """Initialize a RandomForest model."""
    return RandomForestClassifier(random_state=random_state)


def save_model(model, path="../models/model.joblib"):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved at: {path}")


def load_model(path="../models/model.joblib"):
    """Load the trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    model = joblib.load(path)
    print(f"✅ Model loaded from: {path}")
    return model
