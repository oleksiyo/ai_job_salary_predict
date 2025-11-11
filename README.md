
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

