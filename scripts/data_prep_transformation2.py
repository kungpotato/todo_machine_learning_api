import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston

# Load example dataset
data = load_boston()
X, y = data.data, data.target

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Data reduction

# Feature selection
# Select the top k features based on mutual information
k = 5
selector = SelectKBest(mutual_info_regression, k=k)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = data.feature_names[selector.get_support()]

print(f"Selected features (Mutual Information): {selected_features}")

# Dimensionality reduction
# Apply PCA to reduce dimensions
n_components = 2
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_scaled)

print(f"Explained variance ratio (PCA): {pca.explained_variance_ratio_}")
