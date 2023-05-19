#!/usr/bin/python3
import numpy as np
import pandas as pd

df = pd.read_excel("data/data.ods", engine="odf", sheet_name='fillna')

df.fillna(df.median(), inplace=True)
dfFloat = df.copy()
df.info()

for col in df.columns:
    if df[col].dtype == object:
        # print(col)
        values = df[col].unique()
        I = len(values) - 1
        for i, val in enumerate(values):
            # print(val, i / I)
            dfFloat = dfFloat.replace(val, i / I)

# df.fillna(method='ffill', inplace=True)
# print(df)

dfFloat.interpolate(method='linear', limit_direction='backward', inplace=True)
#print(dfFloat)


nulls = df.isnull()
for i in range(len(nulls)):
    if nulls['High Value Customer'][i]:
        if dfFloat['High Value Customer'][i] == .0:
            df['High Value Customer'] = 'No'
        else:
            df['High Value Customer'] = 'Yes'

print(df)

