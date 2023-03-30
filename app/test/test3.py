import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                "exang", "oldpeak", "slope", "ca", "thal", "target"]

data = pd.read_csv(url, header=None, names=column_names, na_values="?")

# Drop rows with missing values
data = data.dropna()

# Preprocess the dataset
X = data.drop("target", axis=1)
y = data["target"].apply(lambda x: 1 if x > 0 else 0)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Predict heart disease for a sample input
sample_input = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
sample_input = scaler.transform([sample_input])
prediction = model.predict(sample_input)
print("Sample input prediction:", "Heart disease" if prediction[0] == 1 else "No heart disease")
