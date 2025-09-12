import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load the dataset
file_path = 'student/student-mat.csv'  # Update this path
data = pd.read_csv(file_path, sep=';')

# Create dropout column
data['dropout'] = data['G3'].apply(lambda x: 1 if x < 10 else 0)

# Separate features and targets
X = data.drop(['G3', 'dropout'], axis=1)  # Features
y_reg = data['G3']                         # Target for regression
y_clf = data['dropout']                    # Target for classification

# List of numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for numeric data
numeric_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split into train/test for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

# Split into train/test for classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42)

# Example: Create preprocessing pipelines
pipeline_reg = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

pipeline_clf = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Fit the pipelines (just for demonstration; models will be added later)
pipeline_reg.fit(X_train_reg, y_train_reg)
pipeline_clf.fit(X_train_clf, y_train_clf)

print("Preprocessing completed. Data is ready for modeling!")
