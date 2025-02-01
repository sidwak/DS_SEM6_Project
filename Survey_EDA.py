import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
print("Encoded DataFrame:\n", df, "\n")
print("Mappings of encoded values back to original:")
for col, encoder in encoders.items():
    print(f"\nColumn: {col}")
    for num, label in enumerate(encoder.classes_):
        print(f"{num} -> {label}")

print(df.head(10))
print(df.loc[0])

print(encoders['Future_Reliance'].classes_)

# Heatmap
plt.figure(figsize=(16,9))
dataplot = sns.heatmap(df.corr(),cmap="YlGnBu", annot=True)
plt.show()

# Bar chart 1
numerical_columns = ['Usage']
grouped_means = df.groupby('Future_Reliance')[numerical_columns].mean()
grouped_means = grouped_means.reset_index()
melted_means = grouped_means.melt(id_vars='Future_Reliance', var_name='Variable', value_name='Mean Value')
plt.figure(figsize=(8, 5))
sns.barplot(data=melted_means, x=encoders['Future_Reliance'].classes_, y='Mean Value', hue='Future_Reliance', palette='viridis')
plt.title('Comparison of Means for Usage by Future_Reliance')
plt.xlabel('Future Reliance')
plt.ylabel('Usage Mean Value')
plt.legend(['Maybe', 'No', 'Yes'],title='Future_Reliance', loc='upper right')
plt.tight_layout()
plt.show()

# Bar chart 2
numerical_columns = ['Delivery_Speed']
grouped_means = df.groupby('Urgent_Needs')[numerical_columns].mean()
grouped_means = grouped_means.reset_index()
melted_means = grouped_means.melt(id_vars='Urgent_Needs', var_name='Variable', value_name='Mean Value')
plt.figure(figsize=(8, 5))
sns.barplot(data=melted_means, x=encoders['Urgent_Needs'].classes_, y='Mean Value', hue='Urgent_Needs', palette='viridis')
plt.title('Comparison of Means for Delivery_Speedage by Urgent_Needs')
plt.xlabel('Urgent Needs')
plt.ylabel('Delivery_Speedage Mean Value')
plt.legend(title='Urgent_Needs', loc='upper right')
plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=df[df['Future_Reliance'] == 0], x='Price_Satisfaction', bins=3, color='blue', label='Future_Reliance = 0', kde=True, stat="density", alpha=0.6)
sns.histplot(data=df[df['Future_Reliance'] == 1], x='Price_Satisfaction', bins=3, color='red', label='Future_Reliance = 1', kde=True, stat="density", alpha=0.6)
sns.histplot(data=df[df['Future_Reliance'] == 2], x='Price_Satisfaction', bins=3, color='green', label='Future_Reliance = 2', kde=True, stat="density", alpha=0.6)
plt.title('Effect of Price_Satisfaction on Future_Reliance Status')
plt.xlabel('Price_Satisfaction')
plt.ylabel('Density')
plt.legend(['Maybe', 'No', 'Yes'],title='Future_Reliance Status', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Piechart
filtered_df = df[df['Unplanned_Items'] == 1]
category_counts = filtered_df['Traditional_Shopping_Replacement'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(category_counts, labels=encoders['Traditional_Shopping_Replacement'].classes_[0:3], autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Traditional Shopping Replacement (Only for Unplanned Items = 1)')
plt.show()

# Line chart
df_sorted = df.sort_values(by='Price_Satisfaction')
df_sorted['Delivery_Speed Rolling Mean'] = df_sorted['Delivery_Speed'].rolling(window=10, min_periods=1).mean()
plt.figure(figsize=(16, 6))
plt.plot(df_sorted['Price_Satisfaction'], df_sorted['Delivery_Speed Rolling Mean'], color='blue', label='Delivery_Speed Trend', linewidth=2)
plt.title('Effect of Price_Satisfaction on Delivery_Speed', fontsize=14)
plt.xlabel('Price_Satisfaction', fontsize=12)
plt.ylabel('Delivery_Speed', fontsize=12)
plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Scatter plot
plt.figure(figsize=(8, 4))
plt.scatter(df['Price_Satisfaction'], df['Delivery_Speed'], color='purple', alpha=0.6, edgecolors='w', s=100)
plt.title('Scatter Plot of Price_Satisfaction vs. Delivery_Speed', fontsize=14)
plt.xlabel('Price_Satisfaction', fontsize=12)
plt.ylabel('Delivery_Speed', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Life_Easier', y='Negative_Impact', data=df, palette='Set2')
plt.title('Comparison of Negative_Impact Across Life_Easier', fontsize=14)
plt.xlabel('Life_Easier', fontsize=12)
plt.ylabel('Negative_Impact', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.xticks([0, 1, 2], encoders['Life_Easier'].classes_, rotation=45)
plt.show()

# Radar Chart
platform_columns = ['Which_Platforms_Blinkit', 
                    'Which_Platforms_SwiggyInstamart', 'Which_Platforms_Zepto']
df_negative = df[df['Negative_Impact'] == 1]
df_non_negative = df[df['Negative_Impact'] == 0]
mean_values_negative = df_negative[platform_columns].mean()
mean_values_non_negative = df_non_negative[platform_columns].mean()
labels = mean_values_negative.index
values_negative = mean_values_negative.values
values_non_negative = mean_values_non_negative.values
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
values_negative = np.concatenate((values_negative, [values_negative[0]]))
values_non_negative = np.concatenate((values_non_negative, [values_non_negative[0]]))
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, values_negative, color='red', alpha=0.25, label="Negative Impact = 1")
ax.plot(angles, values_negative, color='red', linewidth=2)
ax.fill(angles, values_non_negative, color='blue', alpha=0.25, label="Negative Impact = 0")
ax.plot(angles, values_non_negative, color='blue', linewidth=2)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_title("Radar Chart for Platform Usage", size=14)
ax.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()