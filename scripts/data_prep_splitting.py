import numpy as np
from sklearn.model_selection import train_test_split

# Let's assume you have your data in the following format:
# X - features, y - labels
X = np.random.rand(1000, 10) # 1000 samples, 10 features
y = np.random.randint(0, 2, 1000) # 1000 binary labels

# First, we'll split the data into training and the remaining sets (validation + test)
X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=0.4, random_state=42)

# Now, we'll split the remaining data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state=42)

# At this point, we have:
# Training set: X_train, y_train
# Validation set: X_val, y_val
# Test set: X_test, y_test

# Now, you can use these sets to train, tune, and evaluate your machine learning model.
