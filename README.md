# Diabetes Hospital Encounters: EDA and Machine Learning

This project analyses 101,766 hospital encounters of diabetic patients from the public Diabetes 130-US Hospitals dataset. It has three modelling tasks:

- **Regression:** predict how long a patient stays in hospital.
- **Classification:** predict whether diabetes medication is prescribed.
- **Clustering:** group similar patients.

It was an individual university project at Breda University of Applied Sciences (Applied Data Science & AI, Year 1), from November 2024 to January 2025.

## The problem

Length of stay drives bed planning and hospital costs, and patient groups with similar needs can be cared for in similar ways. The project explores how much routinely recorded encounter data can tell us. That data covers demographics, admission type and source, lab procedures, medications and diagnoses.

## How it works

1. **SQL exploration.** Queries on the encounter and patient tables cover:
   - age distribution and missing or anomalous values
   - admission sources and types
   - readmission rates by admission type and by discharge disposition
2. **Python exploration and preprocessing.**
   - Shape, data types and duplicates (there are none).
   - Missing values: `?` is used as the placeholder.
   - Ordinal encoding of age groups and one-hot encoding of categorical columns.
   - Scaling, and train, validation and test splits.
3. **Regression on length of stay** (`time_in_hospital`, in days): linear regression, decision tree, gradient boosting and XGBoost, compared on MAE, MSE and RMSE.
4. **Classification on `diabetesMed`:** logistic regression, random forest and k-nearest neighbours, compared on accuracy, precision, recall and F1.
5. **Clustering:**
   - K-Means with the elbow method, also run on 2-D and 3-D PCA projections.
   - Hierarchical clustering and DBSCAN, compared on silhouette score.
6. **Improvement:**
   - Feature engineering, and feature selection with random forest importances.
   - GridSearchCV for logistic regression and K-Means, and nested cross-validation to check that the tuned settings are stable.
7. **From scratch:** linear and logistic regression with L2 regularisation, implemented in NumPy with vectorised gradient descent.

## Results

**Length of stay** (held-out test set)

| Model | MAE (days) | RMSE (days) |
|---|---|---|
| Linear regression | 2.01 | 2.61 |
| Decision tree | 2.42 | 3.25 |
| Gradient boosting | 1.96 | 2.55 |
| **XGBoost** | **1.95** | **2.55** |

**Diabetes medication prescribed** (77% of encounters are "yes")

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | 0.774 | 0.774 | 1.000 | 0.872 |
| Random forest | 0.628 | 0.807 | 0.683 | 0.740 |
| K-nearest neighbours | 0.727 | 0.782 | 0.897 | 0.836 |

Logistic regression predicts "yes" for every encounter: recall is 1.0 and accuracy equals the share of "yes". No model beat that majority baseline on accuracy. After feature selection and tuning, logistic regression reached 0.774 test accuracy, and 0.782 in nested cross-validation. The recorded encounter features carry little signal about whether medication is prescribed.

**Patient clustering** (silhouette score)

| Method | Silhouette |
|---|---|
| **K-Means** (3 clusters, 2-D PCA) | **0.458** |
| K-Means (3 clusters, all features) | 0.389 |
| Hierarchical | 0.384 |
| DBSCAN | 0.066 |

## My role

This was an individual project; all the analysis and modelling is mine. The course provided the database for the SQL part and the notebook structure.

## Tech stack

Python, SQL, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, seaborn, joblib.

## Repository layout

```
notebooks/
  01_eda_and_modelling.ipynb               # SQL and Python EDA, preprocessing, regression, classification, clustering, tuning
  02_linear_regression_from_scratch.ipynb  # gradient descent with and without L2 regularisation
  03_logistic_regression_from_scratch.ipynb
```

## Data

The dataset isn't included. Download *Diabetes 130-US Hospitals for Years 1999–2008* from the UCI Machine Learning Repository and place it like this (the folders are ignored by git):

```
data/
  diabetic_data.csv      # notebooks 01 and 03
  updated_dataset.csv    # preprocessed, numerically encoded copy used by notebook 02
models/                  # notebook 01 saves the selected models here
```

The SQL queries in notebook 01 are stored as text. They expect the dataset loaded into a relational database with `encounter`, `patient`, `admission_type`, `admission_source` and `discharge_disposition` tables.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```
