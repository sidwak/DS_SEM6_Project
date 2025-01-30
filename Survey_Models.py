import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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


df = pd.read_csv('Impact.csv')
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

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

print("Metrics\n\n")
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

plt.figure(figsize=(16,9))
plot_tree(clf,feature_names=X.columns,class_names=['Yes','Maybe','No'],filled=True,max_depth=3)
plt.title('Decision tree')
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
print(f"Precision (Macro): {precision}")

sensitivity = recall_score(y_test, y_pred, average='macro')
print(f"Sensitivity (Recall, Macro): {sensitivity}")

TN = conf_matrix[0, 0]  # True Negatives
FP = conf_matrix[0, 1]  # False Positives
specificity = TN / (TN + FP)
print(f"Specificity (for class 0): {specificity}")

n = len(y_test)
standard_error = sqrt((precision * (1 - precision)) / n)
print(f"Standard Error of Precision: {standard_error}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# ROC curve for binary classification
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.title('Receiver Operating Characteristic (ROC) Curve - Binary Classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
