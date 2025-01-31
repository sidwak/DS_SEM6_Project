import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori

df = pd.read_csv('Impact.csv')
df = df.drop(['Timestamp','Email Address','Name'],axis=1)

transactions = pd.get_dummies(df[["Which_Platforms", "Product_Categories"]])
frequent_itemsets = apriori(transactions, min_support=0.1, use_colnames=True)
frequent_itemsets
print(frequent_itemsets)
for i, row in frequent_itemsets.iterrows():
    print(row['itemsets'])
    print(row['support'])