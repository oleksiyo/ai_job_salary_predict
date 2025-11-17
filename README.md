# AI Job Salary Prediction — Global AI Job Market 2025

![image](./images/logo.png)


## Problem Description

Over the last few years, the demand for AI professionals has grown rapidly, but the job market remains opaque for both candidates and employers.  
AI engineers, data scientists, and ML practitioners often struggle to answer simple but important questions:

- What salary can I realistically expect given my skills and experience?
- How much do salaries differ between countries, company sizes, and job titles?
- Do remote roles pay more or less than on-site positions?
- Which combinations of experience level, skills, and location are associated with higher pay?

This project uses the **Global AI Job Market & Salary Trends 2025** dataset from Kaggle, containing realistic synthetic data about AI-related job postings.

### 🎯 Goal of the Project

> **Predict the expected salary (`salary_usd`) for AI-related roles based on job characteristics and analyze which features have the strongest impact on pay.**

This includes:

- Performing EDA to understand salary patterns across countries, roles, experience levels, and remote ratio.
- Creating a clean ML pipeline for preprocessing and feature engineering.
- Training multiple regression models and selecting the best-performing one based on RMSE.
- Deploying a lightweight prediction API for practical usage.

### Why This Problem Matters

- Helps **candidates** evaluate whether offers are fair.
- Enables **HR teams** to build competitive and data-driven salary ranges.
- Supports **market analysts** in understanding global salary trends.
- Promotes **transparency** in the global AI job market.

---

## Dataset Description

Source: https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025

The dataset contains **15,000 AI-related job listings** across multiple countries, roles, and company sizes.

### Columns

| Column Name              | Data Type   | Description                                                                                                 |
|--------------------------|-------------|-------------------------------------------------------------------------------------------------------------|
| `job_id`                 | object      | Unique identifier for each job posting                                                                      |
| `salary_usd`             | float64     | Target variable — annual salary in USD                                                                      |
| `salary_currency`        | object      | Original salary currency before normalization                                                               |
| `experience_level`       | object      | Experience level (EN, MI, SE, EX)                                                                           |
| `employment_type`        | object      | Full-time, part-time, contract, internship                                                                  |
| `company_location`       | object      | Country of the employer                                                                                     |
| `company_size`           | object      | Company size category (S/M/L)                                                                               |
| `employee_residence`     | object      | Employee/resident location                                                                                  |
| `remote_ratio`           | int64       | 0/50/100% remote                                                                                            |
| `required_skills`        | object      | Comma-separated list of required skills                                                                     |
| `education_required`     | object      | Minimum required education level                                                                            |
| `years_experience`       | float64     | Required years of experience                                                                                |
| `industry`               | object      | Industry sector                                                                                             |
| `posting_date`           | object      | Posting date                                                                                                |
| `application_deadline`   | object      | Application deadline                                                                                        |
| `job_description_length` | int64       | Length of job description                                                                                   |
| `benefits_score`         | float64     | Normalized benefits score                                                                                   |
| `company_name`           | object      | Hiring company                                                                                              |

---

## Exploratory Data Analysis (EDA)


## Modeling approach & metrics


## How to Run Locally and via Docker

### Run Locally

1. Create virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Start the FastAPI service

```
uvicorn serve:app --reload
```

5. Health check

```
http://127.0.0.1:8000/health

```

Successful response:
```json
{
  "status": "ok"
}
```



### Run with Docker
1. Build the Docker image

```docker
docker build -t ai-salary-api .
```

2. Run the container

```docker
docker run -p 8000:8000 ai-salary-api
```

3. Health check
```
http://localhost:8000/health
```


## API Usage Example
POST /predict

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "job_title": "Machine Learning Engineer",
           "salary_currency": "USD",
           "years_experience": 5,
           "employment_type": "FT",
           "remote_ratio": 100,
           "company_size": "m",
           "company_location": "US",
           "industry": "IT",
           "required_skills": "python, machine learning, deep learning"
         }'
```

Example response:
```json
{
  "predicted_salary_usd": 119744.14
}
```


## Next Steps

## Cloud

python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install xgboost
pip install joblib

pip install wordcloud
pip install uvicorn