# Customer Churn Prediction

Predicting which telecom customers are likely to leave, so the business can act before they do.

## Dataset
IBM Telco Customer Churn dataset (Kaggle): 7,043 customers, 21 columns. After cleaning, 7,032 rows. About 26.6% of customers churned.

## Approach
1. Cleaned data (fixed `TotalCharges` type, removed 11 blank rows).
2. Explored churn patterns with EDA.
3. Encoded categorical features and made an 80/20 stratified train-test split.
4. Trained Logistic Regression, Random Forest and XGBoost, handling class imbalance with class weights.
5. Tuned XGBoost with GridSearchCV (5-fold, scored on ROC-AUC).

## Key Insights
- Customers on month-to-month contracts churn the most.
- Newer customers (low tenure) are more likely to leave.
- Customers with higher monthly charges show a higher churn rate.

## Results
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic (balanced) | 0.493 | 0.781 | 0.605 | 0.835 |
| Random Forest | 0.633 | 0.503 | 0.560 | 0.819 |
| XGBoost | 0.527 | 0.684 | 0.595 | 0.815 |
| **XGBoost (tuned)** | 0.492 | **0.807** | 0.611 | **0.842** |

The tuned XGBoost caught 302 of 374 churners in the test set (recall 0.81). Recall was prioritised over precision because missing a churner costs more than a false alarm. Contract type and internet service type were the top features.

## Tech Stack
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn, Jupyter

## How to Run
1. Download the dataset from Kaggle and place it in the project folder.
2. `pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter`
3. Open `churn_prediction.ipynb` and run all cells.