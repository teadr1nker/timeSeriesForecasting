#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns
from xgboost import XGBClassifier
def toFloat(X):
    XFloat = X.copy()
    for col in X.columns:
        if X[col].dtype == object:
            # print(col)
            values = X[col].unique()
            I = len(values) - 1
            for i, val in enumerate(values):
                # print(val, i / I)
                XFloat = XFloat.replace(val, i / I)
    return XFloat

def importance(frame, removeX = ['Purchased Bike', 'ID'], y = 'Purchased Bike', name='importance'):
    f = toFloat(frame)
    X = f.drop(removeX, axis=1)
    Y = f[y]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)


    model = XGBClassifier()
    model.fit(X_train_scaled, y_train)
    importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.feature_importances_})

    # print(importances)
    plt.bar(importances['Attribute'], importances['Importance'])
    plt.xticks(rotation=45)
    plt.title(name)
    # plt.savefig(name + '.png')
    plt.show()
    plt.clf()

def importanceLogRegression(frame, removeX = ['Purchased Bike', 'ID'], y = 'Purchased Bike', name='importance'):
    f = toFloat(frame)
    X = f.drop(removeX, axis=1)
    Y = f[y]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)


    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    importances = pd.DataFrame(data={
    'Attribute': X.columns,
    'Importance': model.coef_[0]
    })

    # print(importances)
    plt.bar(importances['Attribute'], importances['Importance'])
    plt.xticks(rotation=45)
    plt.title(name)
    # plt.savefig(name + '.png')
    plt.show()
    plt.clf()

df = pd.read_excel("data/data.ods", engine="odf", sheet_name = 'Sheet1')

importance(df)

# columns = []
# for col in df.columns:
#     if df[col].dtype != object and col != 'ID':
#         columns.append(col)
# print(columns)
# print(dfFloat)
X = toFloat(df).drop(['Purchased Bike', 'ID'], axis = 1)
# X = df[columns] # only numerical
print(X)
corrmat = X.corr()
hm = sns.heatmap(corrmat,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 10},
                 yticklabels=X.columns,
                 xticklabels=X.columns,
                 cmap="Spectral_r")
plt.show()
# wcss = []
# for i in range(2, 11):
#     print(f'fitting {i} clusters')
#     kmeans = KMeans(n_clusters=i, random_state=0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
#
# plt.plot(range(2, 11), wcss)
# plt.show()
# quit()

nClusters = 6

kmeans = KMeans(n_clusters=nClusters, random_state=0)
kmeans.fit(X)
pred = kmeans.predict(X)
df['Category'] = pred
print(df.head(16))
df.to_csv('Clusters.csv')
clusters = []
for i in range(nClusters):
    clusters.append(df.loc[pred == i])

for i, cluster in enumerate(clusters):
    print(f'Cluster {i+1}, size {len(cluster)}')
    importance(cluster, ['Purchased Bike', 'ID', 'Category'], name=f'cluster{i} importance')
