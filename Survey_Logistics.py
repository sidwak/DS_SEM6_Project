import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance
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
       'Product_Categories_Groceries','Product_Categories_Householdessentials','Product_Categories_Others','Product_Categories_Snacksandbeverages']]
y = df['Future_Reliance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

model_fi = permutation_importance(model, X, y)
print(model_fi['importances_mean'])

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['Price_Satisfaction'], y=X_test['Delivery_Speed'], hue=y_test, palette={
                0: 'blue', 1: 'red', 2: 'green'}, marker='o')
plt.xlabel("Usage")
plt.ylabel("Delivery_Speed")
plt.title("Logistic Regression Decision Boundary\nAccuracy: {:.2f}%".format(
    accuracy * 100))
plt.legend(['Maybe', 'No', 'Yes'], title="Future_Reliance", loc="upper right")
plt.show()