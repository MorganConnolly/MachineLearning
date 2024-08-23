from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import numpy as np
"""
Trains a neural network on historical housing data to predict the
median house value for California districts.

Utilises scikit-learn's neural network regression API to train the model
with predetermined hyperparameters. 

I completed this project with guidance from the following book:
https://www.amazon.co.uk/Python-Machine-Learning-Example-real-world-dp-1835085628/dp/1835085628/ref=dp_ob_title_bk
"""

# Initialise data sets.
housing = fetch_california_housing()
num_test = 10
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(housing.data[:-num_test, :])
y_train = housing.target[:-num_test].reshape(-1, 1)
X_test = scaler.transform(housing.data[-num_test:, :])
y_test = housing.target[-num_test:]

# Create a neural network with two hidden layers and activation function ReLU.
nn_scikit = MLPRegressor(hidden_layer_sizes = (16, 8), activation = "relu",
                         solver = "adam", learning_rate_init = 0.001,
                         random_state = 42, max_iter = 2000)

nn_scikit.fit(X_train, y_train)

predictions = nn_scikit.predict(X_test) # Testing.
print(predictions)
print(y_test)

# Calculate the mean square error to indicate how effective the regression is.
print(np.mean((y_test - predictions) ** 2))