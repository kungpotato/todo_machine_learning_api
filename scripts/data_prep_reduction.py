import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load example dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing and data reduction pipeline
preprocessing_pipeline = Pipeline(steps=[
    ('scaling', StandardScaler()),  # Scale the features
    ('f_selector', SelectKBest(mutual_info_regression, k=5)),  # Feature selection
    ('pca', PCA(n_components=2))  # Dimensionality reduction
])

# Transform the data
X_train_transformed = preprocessing_pipeline.fit_transform(X_train, y_train)
X_test_transformed = preprocessing_pipeline.transform(X_test)

# Train a machine learning model
model = LinearRegression()
model.fit(X_train_transformed, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_transformed)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
