Predicting Salaries from LinkedIn Job Postings (NLP Case Study)

**Overview**

This project builds an end-to-end natural language processing (NLP) salary prediction model using LinkedIn job postings. The goal is to predict annual salary based on job titles, descriptions, location, work type, and engineered skill signals.

The project emphasizes:

- Real-world data cleaning challenges

- High-dimensional text modeling

- Feature engineering

- Model interpretability

**Dataset**

- Source: LinkedIn Job Postings dataset (via Kaggle)

- Size: ~36,000 postings after cleaning

- Target: Annualized salary (USD)

**Key Challenges Addressed**

- Mixed pay periods (hourly, weekly, monthly, yearly)

- Missing salary values

- Extreme salary outliers

- High-dimensional text features

**Data Cleaning & Preprocessing**
  
Salary Normalization

All salary values were converted to yearly equivalents using standard multipliers:

- Hourly → ×2080

- Weekly → ×52

- Biweekly → ×26

- Monthly → ×12

Target Construction

- salary_target = median_salary when available

- Otherwise: (min_salary + max_salary) / 2

- Rows with no salary information were removed

Outlier Handling

- Salaries above $400,000/year were removed to eliminate data entry errors and extreme noise.

**Feature Engineering**

Text Features

TF-IDF vectorization of:

- Job title

- Job description

Dimensionality reduced using Truncated SVD

**Engineered Features**

- Seniority level inferred from text (intern, junior, mid, senior, lead)

- Description length

- Skill keyword counts, including:

  - Python, SQL, AWS

  - Excel, Tableau, Power BI

  - Machine Learning, Deep Learning

Categorical Features

- Location

- Work type (full-time, temporary, internship)

- Seniority category

**Modeling Approach**

Pipeline

The full pipeline includes:

- Column-wise preprocessing

- TF-IDF vectorization

- One-hot encoding for categorical variables

- Numeric imputation

- Truncated SVD

- ElasticNet regression

**Model Selection**

ElasticNet chosen for:

- Stability in high-dimensional settings

- Handling correlated text features

- Interpretability

Hyperparameters tuned using RandomizedSearchCV

**Model Performance**
Test MAE	~$40K
Test RMSE	~$60K
Test R²	~0.34

Given the inherent noise in salary data and text-based prediction, this represents strong performance.

**Feature Importance & Interpretability**

To interpret the model:

- ElasticNet coefficients were analyzed in the SVD space

- Feature contributions were reconstructed back to original features
  
- Important signals include:

High Salary Signals

- Senior / Director-level titles

- Machine Learning & Deep Learning skills

- High-cost locations (e.g., Washington DC, Austin)

Low Salary Signals

- Associate / Coordinator / Customer Service roles

- Temporary or Internship work types

- Lower-paying metro areas

This aligns closely with real-world labor market patterns.

**Limitations & Future Work**

- TF-IDF does not capture full semantic context (BERT embeddings explored but limited by hardware)

- Company-level effects not modeled

- Salary prediction remains noisy due to market variability

Future improvements could include:

- Transformer-based embeddings

- Company reputation features

- Separate models by job category

**Tools & Libraries**

- Python

- pandas, numpy

- scikit-learn

- matplotlib

- VS Code / Jupyter

**Author**

Mason Ralls
Applied Mathematics (BS)
Aspiring Data Analyst / Data Scientist
