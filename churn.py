import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_excel(r"C:\Users\dell\Downloads\Telco_customer_churn.xlsx")

df.drop(columns=['CustomerID','Count','Country','State'], axis=1, inplace=True)

df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Monthly Charges'] = pd.to_numeric(df['Monthly Charges'], errors='coerce')
df['Total Charges'].fillna(df['Total Charges'].median(), inplace=True)
df['Monthly Charges'].fillna(df['Monthly Charges'].median(), inplace=True)

df['Tenure Months'] = df['Tenure Months'].replace(0, 1)

df['AvgMonthlySpend'] = df['Total Charges'] / df['Tenure Months']
df['HasTechHelp'] = ((df['Online Security'] == 'Yes') | (df['Tech Support'] == 'Yes')).astype(int)
df['HasStreaming'] = ((df['Streaming TV'] == 'Yes') | (df['Streaming Movies'] == 'Yes')).astype(int)
df['HasBundle'] = ((df['Phone Service'] == 'Yes') & (df['Internet Service'] != 'No') & (df['HasStreaming'] == 1)).astype(int)
df['IsLongTermContract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)

target = 'Churn Value'
drop_cols = ['Churn Label', 'Churn Reason']
features = [col for col in df.columns if col not in drop_cols + [target]]

X = df[features]
y = df[target]

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

y_rf_proba = rf_model.predict_proba(X_test)[:, 1]
y_rf_pred = (y_rf_proba >= 0.4).astype(int)

print("Random Forest Accuracy:", accuracy_score(y_test, y_rf_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, y_rf_pred))

joblib.dump(rf_model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_encoded.columns.tolist(), 'model_columns.pkl')
joblib.dump(numerical_cols, 'numerical_columns.pkl')
