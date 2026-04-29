import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Telco Customer Churn.csv")
df.loc[df['TotalCharges'] == ' ', 'TotalCharges'] = 0
df["TotalCharges"] = df["TotalCharges"].astype("float64")

train_columns = [col for col in df.columns if col not in ['customerID', 'Churn']]
X = df[train_columns].copy()
y = df['Churn'].map({'No': 0, 'Yes': 1})

encoders = {}
for col in train_columns:
    le = LabelEncoder()
    # Add a fallback for unseen data by adding a special 'UNSEEN' token if it's object
    X.loc[:, col] = le.fit_transform(X[col])
    encoders[col] = le

clf = LogisticRegression(random_state=42, max_iter=5000)
clf.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump({'model': clf, 'encoders': encoders, 'columns': train_columns}, f)

print("Model saved to model.pkl")
