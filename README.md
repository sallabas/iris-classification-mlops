# iris-classification-mlops

# Sprint 1 – Project Setup & Simple Model  

## Objective  
The goal of **Sprint 1** was to set up the foundational structure of the *Iris Flower Classification* ML project and build a simple working machine-learning model to validate the workflow.  

# Sprint 2 - MLOps Pipeline + API

## Objective 
The goal of **Sprint 2** was to set up a machine-learning pipeline for Iris flower classification built using Kedro, FastAPI, and scikit-learn by following modern MLOps practice. 

---

## Project Setup  
**Repository:** [iris-classification-mlops](https://github.com/sallabas/iris-classification-mlops)  
**Tools & Libraries:** Python 3.12 | scikit-learn | pandas | DVC | Joblib | Jupyter  

### Directory Structure
iris-classification-mlops/
├── irismodelproject/
│   ├── conf/
│   │   └── base/
│   │       ├── catalog.yml
│   │       ├── parameters.yml
│   │       └── parameters_iris_pipeline.yml
│   ├── data/
│   │   ├── 01_raw/
│   │   ├── 02_intermediate/
│   │   ├── 06_models/
│   │   └── 07_model_output/
│   ├── models/
│   │   └── model.joblib
│   ├── src/
│   │   └── irismodelproject/
│   │       ├── pipelines/
│   │       │   └── iris_pipeline/
│   │       │       ├── data_loader.py
│   │       │       ├── model.py
│   │       │       ├── train.py
│   │       │       └── nodes.py
│   │       └── api/
│   │           └── main.py  (FastAPI app)
│   └── notebooks/
│       └── train_model.ipynb
├── requirements.txt
└── README.md

---

## Sprint 1 Outcome  
- Fully functional ML pipeline for Iris classification  
- Data and model artifacts stored in organized directories  
- Code refactored for reusability and future API/pipeline integration  

**Final accuracy:** `1.00`  
**Model file:** `models/model.joblib`  

## Sprint 2 Outcome  
- Designed a full Kedro pipeline covering data integrastion, preprocessing, model training and evaluation
- Configured catalog.yaml for managing dataset across all pipeline stages
- Modularized the ML workflow into nodes
- Enabled automatic model persistence using PickleDataset, storing the trained model under /06_models/
- Achieved 1.00 model accuarcy on the test set within Kedro evaluation step
- Integrated the trained model with FastAPI, enabling real-time predictions
- Exposed API documentation through Swagger UI at // http://127.0.0.1:8000/docs //

---


## Next Steps (Sprint 3 Preview)  
- PyCaret
- Optuna
- Wandb

---

*Created by [@sallabas](https://github.com/sallabas) — Polish-Japanese Academy of Information Technology (PJATK) Project*
