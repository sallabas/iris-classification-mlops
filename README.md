# iris-classification-mlops

# Sprint 1 – Project Setup & Simple Model  

## Objective  
The goal of **Sprint 1** was to set up the foundational structure of the *Iris Flower Classification* ML project and build a simple working machine-learning model to validate the workflow.  

---

## Project Setup  
**Repository:** [iris-classification-mlops](https://github.com/sallabas/iris-classification-mlops)  
**Tools & Libraries:** Python 3.12 | scikit-learn | pandas | DVC | Joblib | Jupyter  

### Directory Structure
iris-classification-mlops/
├── data/
│ └── raw/
│ └── iris.csv
├── models/
│ └── model.joblib
├── notebooks/
│ └── train_model.ipynb
├── src/
│ ├── init.py
│ ├── data_loader.py
│ ├── model.py
│ └── train.py
├── requirements.txt
└── README.md


---

## Sprint 1 Outcome  
- Fully functional ML pipeline for Iris classification  
- Data and model artifacts stored in organized directories  
- Code refactored for reusability and future API/pipeline integration  

**Final accuracy:** `1.00`  
**Model file:** `models/model.joblib`  

---

## Next Steps (Sprint 2 Preview)  
- Build a Kedro pipeline to formalize data and model flow  
- Expose the trained model via a FastAPI endpoint for real-time predictions  

---

*Created by [@sallabas](https://github.com/sallabas) — Polish-Japanese Academy of Information Technology (PJATK) Project*
