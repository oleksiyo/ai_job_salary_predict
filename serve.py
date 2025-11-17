from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load artifacts
with open("model.bin", "rb") as f_in:
    model = pickle.load(f_in)

with open("dv.bin", "rb") as f_in:
    dv = pickle.load(f_in)


# Request schema
class JobFeatures(BaseModel):
    job_title: str
    salary_currency: str
    years_experience: float
    employment_type: str
    remote_ratio: float
    company_size: str
    company_location: str
    industry: str
    required_skills: str


@app.get('/')
def root():
    return {'status': 'OK', 'message': 'AI Job Salary Prediction — Global AI Job Market 2025'}

@app.get('/health')
def health():
    return {"status": "OK"}


@app.post("/predict")
def predict_salary(item: JobFeatures):

    features = item.dict()
    X = dv.transform([features])

    y_pred_log = model.predict(X)[0]
    y_pred = np.expm1(y_pred_log)

    return {"predicted_salary_usd": float(y_pred)}