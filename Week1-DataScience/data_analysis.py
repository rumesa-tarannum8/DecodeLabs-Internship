import pandas as pd

# TASK 1: Load Dataset
df = pd.read_csv("Week1-DataScience/dataset.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# TASK 2: Data Cleaning
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill missing Cabin values
df["Cabin"] = df["Cabin"].fillna("Unknown")

# Remove duplicates
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv("Week1-DataScience/cleaned_dataset.csv", index=False)

print("\nDataset cleaned successfully!")

# TASK 3: Exploratory Data Analysis
print("\n" + "=" * 50)
print("EDA RESULTS")
print("=" * 50)

print("\nStatistical Summary:")
print(df.describe())

print("\nSurvival Count:")
print(df["Survived"].value_counts())

print("\nAverage Age:")
print(df["Age"].mean())

print("\nPassenger Class Distribution:")
print(df["Pclass"].value_counts())

print("\nGender Distribution:")
print(df["Sex"].value_counts())

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# ==========================
# DATA VISUALIZATION
# ==========================

import matplotlib.pyplot as plt

# Survival Count
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.show()

# Passenger Class Distribution
df["Pclass"].value_counts().plot(kind="bar")
plt.title("Passenger Class Distribution")
plt.show()

# Gender Distribution
df["Sex"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.show()

# Age Distribution
df["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.show()

print("\nProject Completed Successfully!")