import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Let's assume you have your data in the following format:
# X - features, y - labels
X = np.random.rand(1000, 10) # 1000 samples, 10 features
y = np.random.randint(0, 2, 1000) # 1000 binary labels

# Define the splitter with 2 splits
splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)

# Perform the first split for training and the remaining sets (validation + test)
for train_index, remain_index in splitter.split(X, y):
    X_train, X_remain = X[train_index], X[remain_index]
    y_train, y_remain = y[train_index], y[remain_index]
    break

# Perform the second split for validation and test sets
for val_index, test_index in splitter.split(X_remain, y_remain):
    X_val, X_test = X_remain[val_index], X_remain[test_index]
    y_val, y_test = y_remain[val_index], y_remain[test_index]
    break

# At this point, we have:
# Training set: X_train, y_train
# Validation set: X_val, y_val
# Test set: X_test, y_test

# Now, you can use these sets to train, tune, and evaluate your machine learning model.
