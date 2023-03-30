import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Example dataset
data = {
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 70000, 80000],
    'gender': ['M', 'F', 'F', 'M'],
    'education': ['High School', 'College', 'Masters', 'PhD']
}

df = pd.DataFrame(data)

# a. Feature scaling
numeric_features = ['age', 'income']
numeric_transformer = StandardScaler()  # Normalize numerical features

# b. Encoding categorical variables
categorical_features = ['gender']
categorical_transformer = OneHotEncoder()  # One-hot encoding for nominal categorical features

ordinal_features = ['education']
ordinal_transformer = OrdinalEncoder(categories=[['High School', 'College', 'Masters', 'PhD']])  # Ordinal encoding for ordinal categorical features

# c. Feature engineering
def add_age_income_interaction(X):
    age_income_interaction = X['age'] * X['income']
    X['age_income_interaction'] = age_income_interaction
    return X

df = add_age_income_interaction(df)

# Combine transformations into a single preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Transform the data
transformed_data = pipeline.fit_transform(df)

print(transformed_data)
