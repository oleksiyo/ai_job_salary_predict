import pickle
import numpy as np


MODEL_PATH = "model.bin"
DV_PATH = "dv.bin"


def load_artifacts():
    with open(MODEL_PATH, "rb") as f_in:
        model = pickle.load(f_in)

    with open(DV_PATH, "rb") as f_in:
        dv = pickle.load(f_in)

    return model, dv


def predict(features: dict):
    model, dv = load_artifacts()

    X = dv.transform([features])
    y_pred_log = model.predict(X)[0]
    y_pred = np.expm1(y_pred_log)

    return float(y_pred)


if __name__ == "__main__":

    example = {
        "job_title": "Machine Learning Engineer",
        "salary_currency": "USD",
        "years_experience": 5,
        "employment_type": "FT",
        "remote_ratio": 100,
        "company_size": "m",
        "company_location": "US",
        "industry": "IT",
        "required_skills": "python, machine learning"
    }

    print("Predicted salary:", predict(example))