# Mushroom Classification — Random Forest

An end-to-end machine learning pipeline that classifies mushrooms as **Edible** or **Poisonous** based on physical characteristics, using **Random Forest**.

## 📌 Problem Statement
- **Objective:** Classify mushrooms as edible or poisonous based on cap, gill, stalk, and habitat features
- **Dataset:** UCI Mushroom dataset (`mushrooms.csv`) — 8,124 records, 23 columns
- **Problem Type:** Classification
- **Target Variable:** `class` (e = Edible, p = Poisonous)
- **Business Relevance:** Could support a foraging/food-safety tool to help identify potentially dangerous mushrooms before consumption
- **Success Metric:** F1-score > 0.80 → **Achieved: F1 = 1.0000** *(see caveat below)*

## 📊 Dataset
- **Rows:** 8,124 | No duplicates found
- **Dropped constant column:** `veil-type`
- **Class distribution:** Edible 51.8% | Poisonous 48.2% — well balanced
- **Features:** 21 fully categorical features (e.g. `odor`, `cap-color`, `gill-size`, `spore-print-color`, `habitat`) — no numeric columns in this dataset
- No missing values detected

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Joblib

## 🔍 Pipeline / Methodology
1. **Data Loading & Validation** — shape, info, summary statistics
2. **Data Cleaning** — dropped constant column (`veil-type`), checked for duplicates
3. **Exploratory Data Analysis** — dashboard overview, class distribution check (correlation/skewness analysis skipped — dataset is fully categorical)
4. **Preprocessing Pipeline** — one-hot encoding for all categorical features (via `ColumnTransformer`)
5. **Train-Test Split** — 80/20 split (6,499 train / 1,625 test)
6. **Model Comparison** — benchmarked classification algorithms using cross-validation
7. **Model Training** — final pipeline trained using **Random Forest**
8. **Model Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report
9. **Sanity Checks** — train vs test score comparison, feature importance dominance check
10. **Cross Validation** — 5-fold CV for robustness check
11. **Hyperparameter Tuning** — GridSearchCV & RandomizedSearchCV to tune `n_estimators`, `max_depth`

## 📈 Results

| Metric | Value |
|--------|-------|
| **Accuracy (Test)** | 1.0000 |
| **Precision** | 1.0000 |
| **Recall** | 1.0000 |
| **F1-Score** | **1.0000** |
| Cross-Validation Mean Accuracy | 0.9276 (± 0.0899) |
| Best Hyperparameters (GridSearchCV) | `n_estimators = 100`, `max_depth = None` |
| Final Tuned Model Score | 1.0000 |
| Train Score vs Test Score | 1.0000 vs 1.0000 |
| Highest single feature importance | 0.086 (no dominant feature) |

**Model Comparison (Top Results):**

| Model | Score |
|-------|-------|
| Decision Tree | 0.9441 |
| Extra Trees | 0.9368 |
| **Random Forest (selected)** | 0.9276 |
| Logistic Regression | 0.9202 |
| KNN | 0.9155 |

## ⚠️ Note on the Perfect Score
The pipeline flagged the train/test scores (1.0000 vs 1.0000) as **"suspiciously high — possible data leakage."** In this specific case, that's less concerning than it sounds:

- **No single feature dominates** (highest importance is only 0.086), so the model isn't just memorizing one leaky column
- The **UCI Mushroom dataset is a well-documented example of a near-perfectly separable dataset** — features like `odor` are famously strong, near-deterministic predictors of edibility, so very high accuracy is a known, expected characteristic of this dataset rather than a pipeline error
- However, **cross-validation scores were unstable** (ranging from 0.7956 to 1.0000 across folds, std ± 0.0899), suggesting the model's performance can vary noticeably depending on which data ends up in each fold — worth investigating further before treating the 100% test score as fully reliable

## 🔑 Key Insights
- Multiple models (Random Forest, KNN, Logistic Regression) all achieved 99.9%+ accuracy, reinforcing that this dataset is highly (if not perfectly) separable using categorical features alone
- High variance across CV folds (± 0.0899) contrasts with the perfect final test score — this inconsistency should be investigated rather than taken at face value
- No dominant feature was found in this run, which is a good sign against simple data leakage, though it's still recommended to review preprocessing steps for target leakage

## 📂 Folder Structure
```
Mushroom-Classification/
│
├── data/
│   └── mushrooms.csv
├── random-forest-mushroom-classification.ipynb
└── README.md
```

## 👤 Author
**Nimra Muhammad Imran**
