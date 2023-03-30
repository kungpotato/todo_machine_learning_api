import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and preprocess the dataset
data = pd.read_csv('../data/your_data.csv')
data = data.dropna()  # Handle missing values
data['categorical_column'] = LabelEncoder().fit_transform(data['categorical_column'])  # Encode categorical variables

# Feature scaling
scaler = StandardScaler()
data[['numerical_column_1', 'numerical_column_2']] = scaler.fit_transform(data[['numerical_column_1', 'numerical_column_2']])

# Split the data into training and validation sets
X = data.drop('target_column', axis=1)
y = data['target_column']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Define candidate models
models = [
    ('Logistic Regression', LogisticRegression()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Support Vector Machines', SVC()),
    ('Decision Trees', DecisionTreeClassifier()),
    ('Neural Networks', MLPClassifier())
]

# Train and evaluate multiple models
best_model = None
best_score = 0.0
for name, model in models:
    model.fit(X_train, y_train)  # Train the model
    predictions = model.predict(X_val)  # Make predictions on the validation set
    score = accuracy_score(y_val, predictions)  # Evaluate model performance
    print(f"{name} - Accuracy: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = model

print(f"Best model: {best_model} - Accuracy: {best_score:.4f} - Model Type: {best_model}")


# Convert the best model to a TensorFlow Keras model
if isinstance(best_model, LogisticRegression):
    model = Sequential()
    model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
elif isinstance(best_model, KNeighborsClassifier):
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
elif isinstance(best_model, SVC):
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
elif isinstance(best_model, DecisionTreeClassifier):
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
elif isinstance(best_model, MLPClassifier):
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
    raise NotImplementedError("Unsupported model type.")

# Train the TensorFlow Keras model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Save the model to a .h5 file
model.save("../app/models/best_model.h5")

