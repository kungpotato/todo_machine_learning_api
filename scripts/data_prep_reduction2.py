import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Feature Selection using Mutual Information
selector = SelectKBest(mutual_info_classif, k=2)  # Select the top 2 features
X_selected = selector.fit_transform(X, y)
selected_features = np.where(selector.get_support())[0]
print(f"Selected Features (using Mutual Information): {selected_features}")

# Dimensionality Reduction using PCA
pca = PCA(n_components=2)  # Reduce dimensions to 2
X_pca = pca.fit_transform(X)
print("Data after PCA transformation:\n", X_pca)

# Dimensionality Reduction using t-SNE
tsne = TSNE(n_components=2)  # Reduce dimensions to 2
X_tsne = tsne.fit_transform(X)
print("Data after t-SNE transformation:\n", X_tsne)

# Dimensionality Reduction using LDA
lda = LDA(n_components=2)  # Reduce dimensions to 2
X_lda = lda.fit_transform(X, y)
print("Data after LDA transformation:\n", X_lda)
