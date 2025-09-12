import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'student/student-mat.csv'  # <-- Change this to your file path
data = pd.read_csv(file_path, sep=';')

# Create dropout column
data['dropout'] = data['G3'].apply(lambda x: 1 if x < 10 else 0)

# --- ADD THE NEW CODE BELOW THIS LINE ---

# Select only numeric columns
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Check correlation with G3
correlation = numeric_data.corr()['G3'].sort_values(ascending=False)
print("Features correlated with final grade (G3):")
print(correlation)

# Plot heatmap of numeric correlations
plt.figure(figsize=(12,8))
sns.heatmap(numeric_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Visualize how attendance affects dropout
plt.figure(figsize=(6,4))
sns.boxplot(x='dropout', y='absences', data=data)
plt.title('Absences vs Dropout')
plt.show()

# Explore how studytime relates to dropout
plt.figure(figsize=(6,4))
sns.countplot(x='studytime', hue='dropout', data=data)
plt.title('Study Time vs Dropout')
plt.show()
