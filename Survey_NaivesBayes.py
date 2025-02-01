import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import graphviz
import pydotplus
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from math import sqrt
import seaborn as sns


df = pd.read_csv('responses.csv')
df = df.drop(['Timestamp','Email Address','Name'],axis=1)

def multi_column_one_hot_encoding(df, columns):
    dummies_list = []
    for col in columns:
        df_cleaned = df[col].str.replace(' ', '').str.get_dummies(sep=',')
        df_cleaned = df_cleaned.add_prefix(f"{col}_")
        dummies_list.append(df_cleaned)

    df = df.drop(columns, axis=1)
    df = pd.concat([df] + dummies_list, axis=1)
    return df

df = multi_column_one_hot_encoding(df, ['Which_Platforms','Product_Categories'])

encoders = {}
encode_cols = ['Age','Gender','Occupation','Usage','Urgent_Needs','Shopping_Planning','Traditional_Shopping_Replacement',
               'Unplanned_Items','Life_Easier','Negative_Impact','Future_Reliance']
#label_encoder = LabelEncoder()
for col in encode_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Print the DataFrame with encoded values
""" print("Encoded DataFrame:\n", df, "\n")
print("Mappings of encoded values back to original:")
for col, encoder in encoders.items():
    print(f"\nColumn: {col}")
    for num, label in enumerate(encoder.classes_):
        print(f"{num} -> {label}") """

print(df.head(10))
print(df.loc[0])

X =df[['Age','Gender','Occupation','Usage','Delivery_Speed','Urgent_Needs','Shopping_Planning','Traditional_Shopping_Replacement',
       'Unplanned_Items','Price_Satisfaction','Life_Easier','Negative_Impact','Which_Platforms_Blinkit',
       'Which_Platforms_FilpkartMinutes','Which_Platforms_SwiggyInstamart','Which_Platforms_SwiggyInstamart','Which_Platforms_Zepto',
       'Product_Categories_Groceries','Product_Categories_Householdessentials','Product_Categories_Others',
       'Product_Categories_Snacksandbeverages']]
y = df['Future_Reliance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print("\nNaives Bayes\n")
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
sns.displot(df, x='Usage', hue='Future_Reliance', kind='kde', fill=True)
sns.displot(df, x='Delivery_Speed', hue='Future_Reliance', kind='kde', fill=True)
plt.legend(['Maybe', 'No', 'Yes'])
plt.show()
