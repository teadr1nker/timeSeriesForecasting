#!/usr/bin/python3
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
# import sklearn as sk
# from sklearn.ensemble import RandomForestClassifier
import scipy.cluster.hierarchy as hcluster

df = pd.read_excel("data/data.ods", engine="odf", sheet_name = 'Sheet1')
dfFloat = df.copy()
df.info()
# for col in df.columns:
#     print(df[col].value_counts())

# print(df.duplicated().sum())
# sns.pairplot(df)
# plt.show()
for col in df.columns:
    if df[col].dtype == object:
        print(col)
        values = df[col].unique()
        I = len(values) - 1
        for i, val in enumerate(values):
            print(val, i / I)
            dfFloat = dfFloat.replace(val, i / I)

print(dfFloat)

# clustering
# thresh = 1000
# clusters = hcluster.fclusterdata(dfFloat.values, thresh, criterion="distance")

# print(len(set(clusters)))
