from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from api.config import load_api_config

config = load_api_config()

# removed gard coded part
app = FastAPI(
    title=config["app"]["title"],
    docs_url=config["app"].get("docs_url", "/docs"),
    openapi_url=config["app"].get("openapi_url", "/openapi.json"),
)

# Optuna
# model = joblib.load("irismodelproject/models/optuna_best_model.pkl")

# Pycaret
model = joblib.load(config["model"]["path"])



class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(data: IrisInput):
    df = pd.DataFrame([{
        "sepal_length": data.sepal_length,
        "sepal_width": data.sepal_width,
        "petal_length": data.petal_length,
        "petal_width": data.petal_width,
    }])

    prediction = model.predict(df)[0]
    confidence = model.predict_proba(df).max()

    return {
        "prediction": int(prediction),
        "confidence": float(confidence)
    }

#    return {
#        "prediction": prediction,
#        "confidence": float(confidence)
#    }

@app.get("/")
def root():
    return {
        "message": "Iris Classification API is running",
        "docs": "/docs",
        "health": "OK"
    }