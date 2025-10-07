# ------------------------------
# Step 0: Import Libraries
# ------------------------------
from google.colab import files
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy import create_engine

uploaded= files.upload()
# ------------------------------
# Step 1: Load CSV Dataset
# ------------------------------
df = pd.read_csv("Customer-Churn-analysis.csv")  # Upload your CSV in Colab

# ------------------------------
# Step 2: Push Raw Data to SQLite (DBT can read this)
# ------------------------------
engine = create_engine("sqlite:///churn_db.sqlite")  # Local SQLite DB
df.to_sql("customer_churn_raw", con=engine, if_exists="replace", index=False)

# ------------------------------
# Step 3: DBT-Like Cleaning (Transform Raw Data)
# ------------------------------
"""
SQL alone cleans data.
🔹 dbt makes those SQL transformations organized, automated, tested, and production-grade.
if u write sql query  
WITH raw AS(...
...
END AS Churn
FROM raw

In dbt, the same thing becomes a model file:
models/staging/stg_customer_churn.sql

And dbt automatically:
Runs it in the right order,
Saves results to a target database,
Documents it,
And lets others reuse it downstream.
So yes — dbt = SQL + structure + automation + testing + versioning 💪

"""
# Simulate what a DBT model would do
query = """
WITH raw AS (
    SELECT * FROM customer_churn_raw
)
SELECT
    customerID,
    CASE WHEN gender='Male' THEN 1 ELSE 0 END AS gender_male,
    CASE WHEN Partner='Yes' THEN 1 ELSE 0 END AS Partner,
    CASE WHEN Dependents='Yes' THEN 1 ELSE 0 END AS Dependents,
    CAST(tenure AS INT) AS tenure,
    CAST(MonthlyCharges AS FLOAT) AS MonthlyCharges,
    CAST(TotalCharges AS FLOAT) AS TotalCharges,
    CASE WHEN PhoneService='Yes' THEN 1 ELSE 0 END AS PhoneService,
    CASE WHEN OnlineSecurity='Yes' THEN 1 ELSE 0 END AS OnlineSecurity,
    CASE WHEN OnlineBackup='Yes' THEN 1 ELSE 0 END AS OnlineBackup,
    CASE WHEN DeviceProtection='Yes' THEN 1 ELSE 0 END AS DeviceProtection,
    CASE WHEN TechSupport='Yes' THEN 1 ELSE 0 END AS TechSupport,
    CASE WHEN StreamingTV='Yes' THEN 1 ELSE 0 END AS StreamingTV,
    CASE WHEN StreamingMovies='Yes' THEN 1 ELSE 0 END AS StreamingMovies,
    CASE WHEN PaperlessBilling='Yes' THEN 1 ELSE 0 END AS PaperlessBilling,
    CASE WHEN Churn='Yes' THEN 1 ELSE 0 END AS Churn
FROM raw
"""
df_cleaned = pd.read_sql(query, con=engine)

# Save cleaned dataset to CSV and SQLite
df_cleaned.to_csv("Customer-Churn-Cleaned.csv", index=False)
df_cleaned.to_sql("customer_churn_cleaned", con=engine, if_exists="replace", index=False)

print("\n✅ Cleaned data saved as 'Customer-Churn-Cleaned.csv' and table 'customer_churn_cleaned' in churn_db.sqlite")


# ------------------------------
# Step 4: Handle Missing Values
# ------------------------------
numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
categorical_cols = ['gender_male','Partner','Dependents','PhoneService','OnlineSecurity',
                    'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                    'StreamingMovies','PaperlessBilling']

# Fill numeric with median
for col in numeric_cols:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

# Fill categorical with mode
for col in categorical_cols:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

# ------------------------------
# Step 4a: Feature Significance Check (Chi-square & T-test)
# ------------------------------
from scipy.stats import chi2_contingency, ttest_ind

significance_results = []

# Categorical features → Chi-square test
for col in categorical_cols:
    if col not in df_cleaned.columns:
        continue
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['Churn'])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        continue
    chi2, p, dof, expected = chi2_contingency(contingency)
    significance_results.append([col, 'Categorical', p])

# Numeric features → T-test
for col in numeric_cols:
    churned = df_cleaned[df_cleaned['Churn']==1][col].dropna()
    not_churned = df_cleaned[df_cleaned['Churn']==0][col].dropna()
    if len(churned) < 2 or len(not_churned) < 2:
        continue
    t_stat, p_value = ttest_ind(churned, not_churned, equal_var=False)
    significance_results.append([col, 'Numeric', p_value])

# Combine results
significance_df = pd.DataFrame(significance_results, columns=['Feature','Type','p_value'])
significance_df['Significant'] = significance_df['p_value'] < 0.05
significance_df = significance_df.sort_values('p_value')

print("\n=== Features Significantly Affecting Churn ===")
print(significance_df[significance_df['Significant']==True])


# ------------------------------
# Step 5: Prepare Features and Target
# ------------------------------
X = df_cleaned[numeric_cols + categorical_cols]
y = df_cleaned['Churn']

# ------------------------------
# Step 6: Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 7: Train Logistic Regression
# ------------------------------
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# Step 8: Predict Churn for Multiple Customers from CSV
# ------------------------------

# Upload CSV of new customers
uploaded = files.upload()  # e.g., new_customers.csv

# Read the uploaded CSV
new_customers = pd.read_csv(list(uploaded.keys())[0])

# Align columns with training data
new_customers = new_customers.reindex(columns=X_train.columns, fill_value=0)

# Predict churn
predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)[:, 1]

# Combine results
results = new_customers.copy()
results['Churn_Prediction'] = predictions
results['Churn_Probability'] = probabilities

# Save predictions to CSV
results.to_csv("churn_predictions.csv", index=False)
print("✅ Predictions saved to 'churn_predictions.csv'")


""" 
# ------------------------------
# Step 8: Predict Churn for a New Customer
# ------------------------------
new_customer = pd.DataFrame({
    'tenure':[12],
    'MonthlyCharges':[70],
    'TotalCharges':[840],
    'gender_male':[1],
    'Partner':[1],
    'Dependents':[0],
    'PhoneService':[1],
    'OnlineSecurity':[0],
    'OnlineBackup':[1],
    'DeviceProtection':[1],
    'TechSupport':[0],
    'StreamingTV':[1],
    'StreamingMovies':[1],
    'PaperlessBilling':[1]
})
"""
# Align columns
new_customers = new_customers.reindex(columns=X_train.columns, fill_value=0)

prediction = model.predict(new_customers)
probability = model.predict_proba(new_customers)[0][1]

print("\nChurn Prediction (1=Will Leave, 0=Will Stay):", prediction[0])
print("Probability of Churn:", round(probability,2))

