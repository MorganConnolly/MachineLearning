import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras 
from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing
import numpy as np
"""
Trains a neural network on historical housing data to predict the
median house value for California districts.

Utilises TensorFlow's Keras API to train the model with predetermined
hyperparameters. 

I completed this project with guidance from the following book:
https://www.amazon.co.uk/Python-Machine-Learning-Example-real-world-dp-1835085628/dp/1835085628/ref=dp_ob_title_bk
"""

# Initialise data sets.
housing = fetch_california_housing()
num_test = 10
scaler = preprocessing.StandardScaler() # Stochastic gradient 
X_train = scaler.fit_transform(housing.data[:-num_test, :])
y_train = housing.target[:-num_test].reshape(-1, 1)
X_test = scaler.transform(housing.data[-num_test:, :])
y_test = housing.target[-num_test:]

# Create a neural network with two hidden layers and activation function ReLU.
# Dropout is used to prevent overfitting.
model = keras.Sequential([
    keras.layers.Dense(units = 20, activation = "relu"),
    keras.layers.Dense(units = 8, activation = "relu"),
    keras.layers.Dense(units = 1),
    tf.keras.layers.Dropout(0.5) 
])
# Uses the Adam optimizer rather than stochastic gradient decent.
model.compile(loss = "mean_squared_error",
              optimizer = tf.keras.optimizers.Adam(0.02)) 
model.fit(X_train, y_train, epochs = 300) # Train 300 iterations.

predictions = model.predict(X_test)[:, 0] # Test
print(predictions)
print(y_test)

# Calculate the mean square error to indicate how effective the regression is.
print(np.mean((y_test - predictions) ** 2))