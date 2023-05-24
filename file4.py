#!/usr/bin/python3
import numpy as np
import pandas as pd

def encode(freq=0):
    return 1 if freq > 0 else 0

df = pd.read_excel("data/data.ods", engine="odf", sheet_name='Shopping')

print(df.info())

def basketAnalysis(col1, col2):
    print(col1, col2)
    df2 = pd.crosstab(df[col1], df[col2])
    #print(df2.head())

    basketInput = df2.applymap(encode)

    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules

    frequentItemsets = apriori(basketInput, min_support=0.001, use_colnames=True)

    rules = association_rules(frequentItemsets, metric="lift")

    rules.sort_values(["support", "confidence", "lift"], axis = 0, ascending = False)
    print(rules.head(16))
    rules.to_csv(f'{col1}_{col2}.csv')


basketAnalysis('Order Number', 'Category')
basketAnalysis('Order Number', 'Product')
