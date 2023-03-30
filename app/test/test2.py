import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
column_names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
df = pd.read_csv(url, header=None, names=column_names, na_values='?')

# Data preprocessing
df.dropna(subset=['price'], inplace=True)
df['normalized-losses'].fillna(df['normalized-losses'].mean(), inplace=True)
df['bore'].fillna(df['bore'].mean(), inplace=True)
df['stroke'].fillna(df['stroke'].mean(), inplace=True)
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
df['peak-rpm'].fillna(df['peak-rpm'].mean(), inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Split the dataset into training and testing sets
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Sample input for prediction
sample_input = {
    'symboling': 3,
    'normalized-losses': 200,
    'wheel-base': 95,
    'length': 150,
    'width': 65,
    'height': 55,
    'curb-weight': 2000,
    'engine-size': 120,
    'bore': 3.6,
    'stroke': 3.0,
    'compression-ratio': 9.0,
    'horsepower': 100,
    'peak-rpm': 5500,
    'city-mpg': 20,
    'highway-mpg': 25,
    'make': 'honda',
    'fuel-type': 'gas',
    'aspiration': 'std',
    'num-of-doors': 'four',
    'body-style': 'sedan',
    'drive-wheels': 'fwd',
    'engine-location': 'front',
    'engine-type': 'ohc',
    'num-of-cylinders': 'four',
    'fuel-system': 'mpfi',
}

# Convert
def create_input(input_dict, df):
    input_df = pd.DataFrame(columns=df.columns)
    for key in input_dict:
        if key in input_df.columns:
            input_df.at[0, key] = input_dict[key]
        else:
            column = f"{key}_{input_dict[key]}"
            if column in input_df.columns:
                input_df.at[0, column] = 1
    input_df.fillna(0, inplace=True)
    return input_df

sample_input_df = create_input(sample_input, X)

predicted_price = model.predict(sample_input_df)
print("Predicted Price:", predicted_price[0])
