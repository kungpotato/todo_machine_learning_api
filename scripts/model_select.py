import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (replace with your own dataset)
data = pd.read_csv('../data/housing_data.csv')

# Extract the number of rooms and prices from the dataset
X = data['rooms'].values.reshape(-1, 1)
y = data['price'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple heuristic: the price of a house is directly proportional to the number of rooms
def simple_heuristic(X):
    return X * 10000  # Here we assume that each room adds $10,000 to the price

# Make predictions using our simple heuristic
y_pred = simple_heuristic(X_test)

# Calculate the mean squared error and R2 score for our simple heuristic
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error: ', mse)
print('R2 score: ', r2)

# Plot the predictions
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Number of rooms')
plt.ylabel('Price')
plt.legend()
plt.show()
