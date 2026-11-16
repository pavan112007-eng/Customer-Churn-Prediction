# Customer Churn Prediction

A classification project that predicts whether a telecom customer will churn, using seven machine learning models and a deep learning model.

## Problem Statement

Churn is a binary classification problem (1 = churned, 0 = retained). Predicting it helps a telecom company identify customers at risk of leaving and plan retention actions.

## Dataset

Customer records from an Iranian telecom company collected over 12 months. The attributes cover the first 9 months; the churn label is the customer's state at month 12.

| Property | Value |
|---|---|
| Rows / columns (raw) | 3,150 / 14 |
| Rows after removing duplicates | 2,850 |
| Target variable | `Churn` (1 = churned, 0 = retained) |
| Class balance | Retained: 2,404 (84.35%), Churned: 446 (15.65%) |

**Key features:** Call Failure, Complains, Subscription Length, Charge Amount, Seconds of Use, Frequency of Use, Frequency of SMS, Distinct Called Numbers, Age Group, Tariff Plan, Status, Customer Value.

## Methodology

1. **Data pre-processing**
   - No missing values.
   - 300 duplicate rows removed (3,150 to 2,850 rows).
   - `Age` dropped, `Age Group` kept; `Tariff Plan` and `Status` recoded from 1/2 to 0/1.
   - Validation: no negative values; categorical fields hold only valid values.
2. **Train-test split:** stratified, `random_state = 42`, across five splits (80-20, 75-25, 70-30, 65-35, 60-40).
3. **Scaling:** `StandardScaler` fit on training data only.
4. **Class imbalance:** SMOTE applied to training data only (80-20: 2,280 to 3,846 rows); the test set keeps the natural class ratio.
5. **Multicollinearity (VIF):** iterative elimination with a threshold of 10 removed `Customer Value` (VIF 47.50) and `Frequency of use` (VIF 18.45), leaving 10 features.
6. **EDA:** target distribution, correlation heatmap, IQR outlier detection, and categorical features vs churn.
7. **Modelling:** 7 ML models and an ANN, evaluated with Accuracy, Precision, Recall, F1 and ROC-AUC.

## Models Compared

Logistic Regression, KNN (k = 5), Decision Tree, Random Forest, SVM, Gradient Boosting, Naive Bayes and a Deep Learning model (ANN).

**ANN setup:** 50 epochs, batch size 32, Adam optimizer, binary cross-entropy loss.

## Results (80-20 split)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 83.86 | 49.04 | 86.52 | 62.60 | 92.48 |
| KNN | 92.46 | 70.18 | 89.89 | 78.82 | 94.77 |
| Decision Tree | 92.46 | 75.00 | 77.53 | 76.24 | 86.35 |
| **Random Forest** | **96.14** | **86.02** | **89.89** | **87.91** | **98.26** |
| SVM | 88.07 | 57.14 | 94.38 | 71.19 | 97.11 |
| Gradient Boosting | 90.53 | 64.23 | 88.76 | 74.53 | 97.18 |
| Naive Bayes | 65.96 | 30.34 | 91.01 | 45.51 | 89.25 |
| Deep Learning (ANN) | 95.09 | 82.80 | 86.52 | 84.62 | - |

**Best model: Random Forest** (96.14% accuracy, 87.91% F1, 98.26% ROC-AUC). It also gave the highest F1 on the 80-20 split. The ANN came second on F1 (84.62%), and Naive Bayes was the weakest (45.51%).

## Key Findings

- **Complaints:** 82.61% churn for customers who complained vs 9.77% for those who did not.
- **Status:** 47.51% churn for non-active customers vs 5.59% for active ones.
- **Tariff plan:** 16.79% churn for pay-as-you-go customers vs 2.62% for contractual customers.
- **Correlation with churn:** Complains (0.55) and Status (0.49) are the strongest positive correlations; Frequency of use and Seconds of Use (-0.30) are the strongest negative ones.

## Business Applications

- Flag customers likely to churn and target them with retention offers.
- Prioritise customers who have complained.
- Monitor non-active customers.
- Focus retention efforts on pay-as-you-go customers.

## Limitations and Future Work

- SMOTE creates synthetic samples by interpolation, which is not ideal for binary and ordinal features.
- The data comes from a single telecom company over 12 months.
- Future work: more attributes and customers, and further tuning of the best models.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE), TensorFlow/Keras, Matplotlib, Seaborn

## Repository Contents

- `Customer Churn Prediction PPT.pptx`: project presentation with full results and appendix

## Team

- Thota Ritvika Reddy
- Chivukula Pavan Kalyan
- Pullapally Amulya Yadav
- Reddaboina Ajay Kumar
