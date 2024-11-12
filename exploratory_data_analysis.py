# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:53:28 2024

@author: Ashish
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew



# Load the dataset
data = pd.read_csv('train.csv')

# Display the first few rows of the dataset
print(data.head())

# Analyze the data for missing values
missing_data = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_data)

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show(block=False) 
plt.pause(2) 
plt.close()

# Percentage of missing values in each column
missing_percentage = (missing_data / len(data)) * 100
print("\nPercentage of missing values in each column:")
print(missing_percentage)

# Handling missing values
# Strategy: Fill numerical columns with the mean and categorical columns with the mode

# Fill missing values in numerical columns with the mean
# Data Cleaning
# Fill missing values based on the type and distribution of data
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Work Pressure'] = data['Work Pressure'].fillna(data['Work Pressure'].mean())
data['Academic Pressure'] = data['Academic Pressure'].fillna(data['Academic Pressure'].mean())
data['CGPA'] = data['CGPA'].fillna(data['CGPA'].median())
data['Job Satisfaction'] = data['Job Satisfaction'].fillna(data['Job Satisfaction'].mode()[0])
data['Study Satisfaction'] = data['Study Satisfaction'].fillna(data['Study Satisfaction'].mode()[0])

# Fill missing values in categorical columns with the mode
categorical_cols = data.select_dtypes(include=[object]).columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)


# Convert relevant columns to categorical type:  
# converting each column in the categorical_features list to the category data type in a pandas DataFrame.
categorical_features = ['Gender', 'City', 'Working Professional or Student', 
                        'Profession', 'Degree', 'Dietary Habits', 'Sleep Duration', 
                        'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
for col in categorical_features:
    data[col] = data[col].astype('category')



# Verify if missing values have been handled
print("\nMissing values after handling:")
print(data.isnull().sum())

# Save the cleaned dataset
data.to_csv('cleaned_mental_health_data.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_mental_health_data.csv'.")

# Use the cleaned dataset to compute summary statistics
cleaned_data = pd.read_csv('cleaned_mental_health_data.csv')

# Summary statistics of the cleaned dataset
print("\nSummary statistics of the cleaned dataset:")
print(cleaned_data.describe(include='all'))


######### EDA on cleaned Dataset  #################

# Load Cleaned Dataset
data = pd.read_csv('cleaned_mental_health_data.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Dataset Overview
print("\nDataset Information:")
print(data.info())
print("\nSummary Statistics:\n", data.describe())

# Checking for any remaining missing values
print("\nMissing Values Count:")
print(data.isnull().sum())

# Step 1: Univariate Analysis
# ---------------------------
# Plot distributions for numerical features
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True, color="skyblue", stat="density", linewidth=0)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.show(block=False) 
    plt.pause(2) 
    plt.close()

# Plot boxplots for numerical features to check for outliers
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[col], color="lightcoral")
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show(block=False) 
    plt.pause(2) 
    plt.close()

# Plot count plots for categorical features
categorical_features = data.select_dtypes(include='category').columns
for col in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=data[col], palette="viridis")
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.show(block=False) 
    plt.pause(2) 
    plt.close()

# Step 2: Bivariate Analysis
# --------------------------
# Relationship with Target Variable (Depression)
target = 'Depression'
for col in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, hue=target, data=data, palette="coolwarm")
    plt.title(f'{col} vs {target}')
    plt.xticks(rotation=45)
    plt.show(block=False) 
    plt.pause(2) 
    plt.close()

# Bivariate analysis for numerical features with the target variable
for col in numerical_features:
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=data, x=col, hue=target, fill=True, common_norm=False, palette="Set1")
    plt.title(f'Distribution of {col} by {target}')
    plt.xlabel(col)
    plt.show(block=False) 
    plt.pause(5) 
    plt.close()

# Step 3: Multivariate Analysis
# -----------------------------
# Correlation Heatmap for numerical variables
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix for Numerical Features")
plt.show(block=False) 
plt.pause(5) 
plt.close()

# Pairplot for numerical features with hue as the target variable
sns.pairplot(data, hue=target, palette="coolwarm", diag_kind='kde', corner=True)
plt.show(block=False) 
plt.pause(5) 
plt.close()

# Step 4: Feature Insights and Summary Statistics
# ----------------------------------------------
# Categorical features summary
for col in categorical_features:
    print(f"\nSummary of {col}:")
    print(data[col].value_counts())

# Skewness and Kurtosis for numerical features
print("\nSkewness and Kurtosis for Numerical Features:")
for col in numerical_features:
    skewness = skew(data[col].dropna())
    kurtosis = data[col].kurtosis()
    print(f"{col}: Skewness = {skewness:.2f}, Kurtosis = {kurtosis:.2f}")

# Saving Key EDA Figures
# ----------------------
# Save pairplot for documentation
pairplot_fig = sns.pairplot(data, hue=target, palette="coolwarm", diag_kind='kde', corner=True)
pairplot_fig.savefig("pairplot.png")

# Save correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix for Numerical Features")
plt.savefig("correlation_heatmap.png")

print("\nEDA Completed. Key plots saved as 'pairplot.png' and 'correlation_heatmap.png'.")