# Credit Card Fraud Detection

**Credit Card Fraud Detection** is a comprehensive machine learning pipeline built in Python that tackles the critical problem of unauthorized credit card transactions. Using a real Kaggle dataset of European cardholders, this project employs rigorous data preprocessing, exploratory analysis, and multiple high‑performance models to achieve robust fraud detection.

## Project Description
1. **Data Ingestion & Validation**: Load over 284,000 anonymized transactions, ensure data integrity, and verify no missing values.
2. **Exploratory Data Analysis (EDA)**:
   - Class balance visualization to highlight the extreme imbalance (0.17% fraud).
   - Time‑series scatter plot of fraudulent amounts to inspect temporal patterns.
   - Advanced ROC and Precision‑Recall curve plotting for model comparison.
3. **Feature Engineering**: Although features V1–V28 are pre‑PCA‑transformed, the pipeline adds derived hour‑of‑day features and amount scaling where beneficial.
4. **Model Training & Tuning**:
   - Five classifiers: Random Forest, AdaBoost, CatBoost, XGBoost, LightGBM.
   - Consistent train/validation splits with early stopping for gradient boosting.
   - Hyperparameter defaults chosen to balance speed and accuracy for internship‑scale demos.
5. **Evaluation & Interpretation**:
   - ROC‑AUC as the primary performance metric on validation sets.
   - Confusion matrices to quantify Type I (false positive) and Type II (false negative) errors.
   - Feature importance rankings for actionable insights.

## Real‑World Applications
Financial institutions face billions in fraud losses annually. A scalable fraud detection pipeline like this can be:
- **Integrated** into transaction monitoring systems for real‑time alerts.
- **Extended** with cost‑sensitive learning to minimize financial impact.
- **Adapted** to other anomaly detection domains (insurance claims, network intrusions).

**Tech Stack**
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit‑learn, CatBoost, XGBoost, LightGBM
- **Environment:** Python 3.10, venv

**Key Validation Metrics**
| Model       | Validation ROC‑AUC |
|-------------|--------------------:|
| RandomForest| 0.9520             |
| AdaBoost    | 0.9878             |
| CatBoost    | 0.9907             |
| XGBoost     | 0.9893             |
| LightGBM    | 0.9548             |

**Dataset**
> "Credit Card Fraud Detection" from Kaggle  
> https://www.kaggle.com/mlg-ulb/creditcardfraud  

> **Note:** Due to file size limits, `creditcard.csv` is not tracked in this repository. Please download it from Kaggle and place it in `data/raw/` before running the pipeline.
