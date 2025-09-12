import pandas as pd

# Load the dataset
file_path = 'student/student-mat.csv'
data = pd.read_csv(file_path, sep=';')

# Show the first few rows
print(data.head())

# Show summary information
print(data.info())

# Show basic statistics
print(data.describe())
