import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# # Drop rows with missing values
# cleaned_data = data.dropna()

# # Drop columns with missing values
# cleaned_data = data.dropna(axis=1)

# # Drop rows where specific columns have missing values
# cleaned_data = data.dropna(subset=['column1', 'column2'])

# # Fill missing values with 0
# filled_data = data.fillna(0)

# # Fill missing values with a custom value
# filled_data = data.fillna(-999)


# # Fill missing values with the mean
# mean_value = data['column'].mean()
# filled_data = data.fillna(mean_value)

# # Fill missing values with the median
# median_value = data['column'].median()
# filled_data = data.fillna(median_value)

# # Fill missing values with the mode
# mode_value = data['column'].mode().iloc[0]
# filled_data = data.fillna(mode_value)

# # Linear interpolation
# interpolated_data = data.interpolate()

# # Interpolation with different methods (e.g., 'polynomial', 'spline')
# interpolated_data = data.interpolate(method='polynomial', order=2)


# Sample dataset with missing values
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5]
})

# Impute missing values using KNNImputer
imputer = KNNImputer(n_neighbors=2)
imputed_data = imputer.fit_transform(data)

# Convert the imputed data back to a pandas DataFrame
imputed_data = pd.DataFrame(imputed_data, columns=data.columns)
print(imputed_data)


# Sample dataset with missing values
data2 = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [1, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5]
})

# Split the data into two parts: with and without missing values
data_with_missing_values = data2[data2.isna().any(axis=1)]
data_without_missing_values = data2.dropna()

# Train a linear regression model to predict the missing value in column 'C'
X_train = data_without_missing_values[['A', 'B']]
y_train = data_without_missing_values['C']
X_test = data_with_missing_values[['A', 'B']]

# Fit the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the missing value and fill it in the DataFrame
predicted_value = regressor.predict(X_test)
data.loc[data['C'].isna(), 'C'] = predicted_value

print(data2)


# These examples demonstrate how to use
# machine learning models to impute missing values
# in a dataset. The choice of which model to use 
# depends on the nature of the data and
# the relationships between features.
# You may need to experiment with different models
# and techniques to find the best approach for
# your specific problem.
