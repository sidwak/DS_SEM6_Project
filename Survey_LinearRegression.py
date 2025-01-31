import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


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

X = df[['Delivery_Speed']]
y = df['Price_Satisfaction'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()

model.fit(X_train, y_train)

print(f"Coefficient: {model.coef_[0]}") # Coefficients
print(f"Intercept: {model.intercept_}") # intercept

main_pred = model.predict(X_test)


plt.figure(figsize=(8, 6))
#sns.scatterplot(x='age', y='adiposity', data=df, color='blue', label='Data Points')
plt.plot(X_test, main_pred, color='green', label='Regression Line')
plt.plot(y_test, main_pred, color='red', label='Predicted Line')
#sns.scatterplot(x='age', y='adiposity', data=pd.DataFrame({'age': age_pred, 'adiposity': y_pred}), color='orange', label='Data Points')
plt.xlabel('Delivery_Speed')
plt.ylabel('Price_Satisfaction')
plt.title('Linear Regression: Price_Satisfaction vs Delivery_Speed')
plt.legend()
plt.show()

mse = mean_squared_error(main_pred, y_test)  # Mean Squared Error
mae = mean_absolute_error(main_pred, y_test)  # Mean Absolute Error
r2 = r2_score(main_pred, y_test)  # R-squared (R²)

# Output the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")


