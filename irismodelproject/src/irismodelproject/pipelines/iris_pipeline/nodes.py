import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def load_data(data: pd.DataFrame):
    X = data.drop("species", axis=1)
    y = data["species"]
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=10)


def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=10)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    return accuracy


def save_model(model):
    joblib.dump(model, "models/model.joblib")

