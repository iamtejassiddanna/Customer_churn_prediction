import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def train_and_save():
    df = pd.read_csv("Telco Customer Churn.csv")
    
    # Preprocessing
    df.loc[df['TotalCharges'] == ' ', 'TotalCharges'] = 0
    df["TotalCharges"] = df["TotalCharges"].astype("float64")

    # Features and Target
    train_columns = [col for col in df.columns if col not in ['customerID', 'Churn']]
    X = df[train_columns].copy()
    y = df['Churn'].map({'No': 0, 'Yes': 1})

    encoders = {}
    
    for col in train_columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            # Adding an 'UNSEEN' class for robustness
            classes = list(X[col].unique()) + ['UNSEEN']
            le.fit(classes)
            X.loc[:, col] = le.transform(X[col])
            encoders[col] = le
        else:
            # Leave numerical columns as they are
            pass

    # Model definition
    clf = LogisticRegression(random_state=42, max_iter=5000)
    clf.fit(X, y)

    # Save model and encoders
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': clf, 'encoders': encoders, 'columns': train_columns}, f)

    print("Model trained and saved to model.pkl successfully.")

if __name__ == '__main__':
    train_and_save()
