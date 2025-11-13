
![image](./images/logo.png)

## Problem description
Over the last few years, the demand for AI professionals has grown rapidly, but the job market remains opaque for both candidates and employers.  
AI engineers, data scientists and ML practitioners often struggle to answer simple but important questions:
- What salary can I realistically expect given my skills and experience?
- How much do salaries differ between countries, company sizes, and job titles?
- Do remote roles pay more or less than on-site positions?
- Which combinations of experience level, skills, and location are associated with higher pay?


The project uses the [Global AI Job Market & Salary Trends 2025](https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025?) dataset from Kaggle, which contains synthetic but realistic data about AI-related jobs: job titles, experience level, salary (USD and local), location, company size, remote ratio, benefits score, required skills, education level, and other features.

The goal is to solve the following problem:

> Predict the expected salary (salary_usd) for a job in the AI field based on its characteristics and analyze which factors have the biggest impact on pay.

This includes:
- performing EDA to understand salary patterns across roles, countries, experience levels, and remote ratio;
- training a regression model that predicts salary based on job attributes.

The final solution can help users explore AI job-market trends and estimate salary ranges for different job profiles.


## Data Specification

https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025?resource=download

Columns:
| Column Name              | Data Type   | Description                                                                                                 |
| ------------------------ | ----------- | ----------------------------------------------------------------------------------------------------------- |
| `job_id`      	       | object      | Unique identifier for each job posting                                                             
| `salary_usd`             | float64     | Annual salary in USD for the AI/ML job listing.                                                             |
| `salary_currency`        | object      | Currency of the reported salary before conversion to USD.                                                   |
| `experience_level`       | object      | Level of experience required (e.g., “EN” = Entry-level, “MI” = Mid-level, “SE” = Senior, “EX” = Executive). |
| `employment_type`        | object      | Type of employment contract (e.g., full-time, part-time, contract).                                         |
| `company_location`       | object      | Country where the hiring company is located.                                                                |
| `company_size`           | object      | Size category of the company (S = Small, M = Medium, L = Large).                                            |
| `employee_residence`     | object      | Country where the employee resides (or is expected to reside) for the job.                                  |
| `remote_ratio`           | int64       | Percentage of remote work allowed (0 = on-site, 50 = hybrid, 100 = fully remote).                           |
| `required_skills`        | object      | List or string of skills required for the job (e.g., Python, SQL, TensorFlow).                              |
| `education_required`     | object      | Minimum education level required (e.g., Bachelor, Master, PhD).                                             |
| `years_experience`       | float64/int | Years of experience required for the job.                                                                   |
| `industry`               | object      | Industry sector of the job posting (e.g., Technology, Healthcare, Finance).                                 |
| `posting_date`           | object/date | Date when the job was posted.                                                                               |
| `application_deadline`   | object/date | Deadline date for job applications.                                                                         |
| `job_description_length` | int64       | Length (in characters or words) of the job description text.                                                |
| `benefits_score`         | float64/int | A numeric score indicating how many/which benefits are offered (e.g., health, remote, bonus).               |
| `company_name`           | object      | Name of the hiring company.                                                                                 |

