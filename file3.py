#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def findOutliers(df):
    q1=df.quantile(0.3)

    q3=df.quantile(0.7)

    IQR=q3-q1

    outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
    return outliers



df = pd.read_excel("data/data.ods", engine="odf", sheet_name='Sheet1')

print(df.describe())

findOutliers(df)
for col in df.columns:
    if df[col].dtype != 'object':
        outliers = findOutliers(df[col])
        print(f'\n\n {col}')
        print("number of outliers:" + str(len(outliers)))

        print("max outlier value:" + str(outliers.max()))

        print("min outlier value:" + str(outliers.min()))
