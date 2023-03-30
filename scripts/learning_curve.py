import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Define the logistic regression model
model = LogisticRegression(solver='liblinear')

# Compute the learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy"
)

# Calculate the mean and standard deviation of the training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot the learning curve
plt.figure()
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')

plt.title('Learning Curve')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()

# This code will generate a learning curve plot
# showing the model's training and validation scores
# as the number of training examples increases. 
# The shaded areas around the curves represent 
# the standard deviation of the scores.

# If the validation curve continues to rise 
# as the number of training examples increases,
# it's an indication that adding more data 
# could lead to improved model performance.
# If the validation curve plateaus or flattens,
# it suggests that adding more data might not have
# a significant impact on the model's performance.
# In such cases, you may consider using other 
# techniques to improve the model, such as feature engineering,
# hyperparameter tuning, or trying a different model architecture.