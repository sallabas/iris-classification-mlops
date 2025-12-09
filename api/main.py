from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Modeli yükle
model = joblib.load("irismodelproject/data/06_models/model.pkl")

app = FastAPI(title="Iris Classification API")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisInput):
    arr = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(arr)[0]
    return {"prediction": int(prediction)}
