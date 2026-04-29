import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---

# importing the csv file
df = pd.read_csv("Telco Customer Churn.csv")

# ---

df.head()

# ---

df.columns

# ---

df.info()

# ---

# grouping the same data type columns
df.columns.to_series().groupby(df.dtypes).groups

# ---



# ---

# exploring all the values of features
for elem in df.columns [1:]:
    print(df[elem].value_counts())
    print('\n')

# ---

df['TotalCharges'].value_counts()        

# ---

# preprocessing the TotalCharges, changing the null values with zero

df.loc[ df['TotalCharges'] == ' ', 'TotalCharges'] = 0
df["TotalCharges"] = df["TotalCharges"].astype("float64")

# ---

# checking the null values
df['TotalCharges'].isna().sum()

# ---

# preprocessing the senior citizens, converting the int64 to object
df['customerID'] = df['customerID'].astype('object')

# ---

df.isna().sum()

# ---

df.describe()

# ---

df['Churn'].value_counts().plot(kind = 'bar')

# ---

# exploring the numerical one by one
fig, ax = plt.subplots(6, figsize=(10, 15))
sns.set(font_scale=0.5)

g=sns.boxplot(data=df, x="tenure", ax=ax[0])
g=sns.histplot(data=df, x="tenure", ax=ax[1])


g=sns.boxplot(data=df, x="MonthlyCharges", ax=ax[2])
g=sns.histplot(data=df, x="MonthlyCharges", ax=ax[3])


g=sns.boxplot(data=df, x="TotalCharges", ax=ax[4])
g=sns.histplot(data=df, x="TotalCharges", ax=ax[5])

# ---



# ---

# exploring churn with other numerical variables
fig, ax = plt.subplots(6, figsize = (15,23))
sns.set(font_scale =1.5)

g = sns.boxplot(data = df, x= "tenure", y= "Churn", ax= ax[0], hue="Churn") 
g = sns.histplot(data = df,x= "tenure", ax=ax[1], hue= "Churn")

g = sns.boxplot(data = df, x= "MonthlyCharges", y= "Churn", ax=ax[2], hue= "Churn")
g = sns.histplot(data = df, x= "MonthlyCharges",ax= ax[3], hue="Churn")

g = sns.boxplot(data = df, x= "TotalCharges", y= "Churn", ax=ax[4], hue= "Churn")
g = sns.histplot(data = df, x= "TotalCharges",ax= ax[5], hue="Churn")

# ---

# exploring the categorial vaiables
categorical_variables = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod']


# ---

# converting the categorical features (non numeric) - one hot encoding
df_dummies = pd.get_dummies(df[categorical_variables])

# adding the numerical values
df_dummies[['tenure', 'MonthlyCharges', 'TotalCharges']] = df[['tenure', 'MonthlyCharges', 'TotalCharges']]

df_dummies['Churn']= df['Churn']
df_dummies['Churn'] = df_dummies['Churn'].map({'No': 0, 'Yes': 1})
df_dummies.tail()

# ---

#Get Correlation of "Churn" with other variables:
sns.set(font_scale=1)
plt.figure(figsize=(15,8))
df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')

# ---

sns.set(font_scale=2)
fig, ax = plt.subplots(figsize=(30,30))

# Generating a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

mask = np.triu(np.ones_like(df_dummies.corr(), dtype=bool))
sns.heatmap(df_dummies.corr(), mask=mask, cmap=cmap)

# ---



# ---

# machine learning models

# ---

#prepare for model training
train_columns = df.columns.to_list()

for elem in ["customerID","Churn"]:
    train_columns.remove(elem)

# ---

# converting the categorical features
df_dummies = pd.get_dummies(df[categorical_variables])

# adding the numerical features
df_dummies[['tenure', 'MonthlyCharges', 'TotalCharges']] = df[['tenure', 'MonthlyCharges', 'TotalCharges']]

df_dummies['Churn'] = df['Churn']
df_dummies['Churn'] = df_dummies['Churn'].map({'No': 0, 'Yes': 1})


y = df_dummies['Churn']
X = df[train_columns].copy()


from sklearn.preprocessing import LabelEncoder

for col in train_columns:

    le = LabelEncoder()
    le.fit(X[col].to_list())
    X.loc[:, col] = le.transform(X[col])


clf_stats_df = pd.DataFrame(columns=["clf_name", "F1-score", "auc-score", "elapsed_time"])
roc_auc_score_list = []

# ---

# logistic regression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import time
import seaborn as sns


# splittig data for train test 
xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)

from sklearn.linear_model import LogisticRegression

start_time = time.time()

predictions_probas_list = np.zeros([len(yvalid), 2])



roc_auc_list = []
num_of_folds = 10
num_fold = 0

#feature_importance_df = pd.DataFrame()

folds = StratifiedKFold(n_splits=num_of_folds, shuffle=False)

for train_index, test_index in folds.split(xtrain, ytrain):
    xtrain_stra, xtest_stra = xtrain.iloc[train_index,:], xtrain.iloc[test_index,:]
    ytrain_stra, ytest_stra = ytrain.iloc[train_index], ytrain.iloc[test_index]

    print()
    print("Stratified Fold:", num_fold)
    num_fold = num_fold + 1
    print()

    clf_stra_logit = LogisticRegression(random_state=42, max_iter=5000)

    clf_stra_logit.fit(xtrain_stra, ytrain_stra)


    predictions = clf_stra_logit.predict(xtest)
    predictions_probas = clf_stra_logit.predict_proba(xtest)
    predictions_probas_list += predictions_probas/num_of_folds

    roc_auc_list.append(roc_auc_score(ytest, predictions_probas[:,1], average = "macro"))


predictions = np.argmax(predictions_probas_list, axis=1)
roc_auc_score_list.append(roc_auc_list)

print()
print(classification_report(ytest, predictions))

print()
print("f1_score", f1_score(ytest, predictions, average = "macro"))

print()
print("roc_auc_score", roc_auc_score(ytest, predictions_probas_list[:,1], average = "macro"))

elapsed_time = time.time() - start_time
clf_stats_df = pd.concat([clf_stats_df, pd.DataFrame([{"clf_name": "clf_stra_logit",
                     "F1-score":f1_score(ytest, predictions, average = "macro"),
                     "auc-score": roc_auc_score(ytest, predictions_probas_list[:,1], average = "macro"),
                     "elapsed_time": elapsed_time}])], ignore_index=True)

print()
print("elapsed time in seconds: ", elapsed_time)
print()
import gc
gc.collect();

# ---



# ---

