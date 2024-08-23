import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing
"""
Trains a neural network on historical housing data to predict the
median house value for California districts.

The neural network has one layer and training is completed by adjusting
the weight matrices and bias vectors using backpropagation.

I completed this project with guidance from the following book:
https://www.amazon.co.uk/Python-Machine-Learning-Example-real-world-dp-1835085628/dp/1835085628/ref=dp_ob_title_bk
"""

def sigmoid(z):
    """
    Activation function.
    """
    return 1.0 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """
    Derivative of the activation function utilised in backtracking.
    """
    return sigmoid(z) * (1.0 - sigmoid(z))

def train(X_train, y_train, n_hidden_neurons, learning_rate, n_iter):
    '''
    Creates a neural network with one hidden layer.

    Weights and biases are initially random. Each iteration we calculate
    the activation function for the hidden & output vectors using the
    existing weights and biases. Resulting gradients are calcualted and
    weights & biasses are reassigned.
    @param X_train: np array, training feature data
    @param y_train: np array, training target data
    @param n_hidden_neurons: int, number of neurons in the hidden layer
    @param learning_rate: float, rate which the weights & biases are
    changed.
    @param n_iter: n, num of iterations
    @return: {W1: 1st set of weights, b1: 1st set of biases,
              W2: 2nd set of weights, b2: 2nd set of biases}
    '''
    n_samples, n_features = X_train.shape 
    W1 = np.random.randn(n_features, n_hidden_neurons) # 1st weight matrix. Initially random.
    b1 = np.zeros((1, n_hidden_neurons))               # hidden vector bias. Initially all zeros.
    W2 = np.random.randn(n_hidden_neurons, 1)          # 2nd weight matrix. Initially random.
    b2 = np.zeros((1,1))                               # output vector bias. Initially all zeros.
    for i in range(1, n_iter + 1):
        Z2 = np.matmul(X_train, W1) + b1                    # matrix multiplication before activation function applied.
        A2 = sigmoid(Z2)                                    # hidden vector w/ activation function.
        Z3 = np.matmul(A2, W2) + b2                         # matrix multiplication before activation function applied.
        A3 = Z3                                             # output vector.
        dZ3 = A3 - y_train                                  # derivative of the cost function for input to the output layer.
        dW2 = np.matmul(A2.T, dZ3)                          # gradient of the cost of 2nd weight matrix.
        db2 = np.sum(dZ3, axis = 0, keepdims = True)        # sum of dZ3
        dZ2 = np.matmul(dZ3, W2.T) * sigmoid_derivative(Z2) # derivative of the cost function for input to the hidden layer.
        dW1 = np.matmul(X_train.T, dZ2)                     # gradient of the cost of 1st weight matrix.
        db1 = np.sum(dZ2, axis = 0)                         # sum of dZ2
        W2 = W2 - learning_rate * dW2 / n_samples 
        b2 = b2 - learning_rate * db2 / n_samples 
        W1 = W1 - learning_rate * dW1 / n_samples 
        b1 = b1 - learning_rate * db1 / n_samples 

        if (i % 100) == 0:
            cost = np.mean((y_train - A3) ** 2)
            print(f"Iteration {i}, training loss: {cost}")
    
    model = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return model

def predict(test_data, model):
    '''
    Given a model and test data, calculates the value of hidden & output
    nodes for easy comparison against test data target values.
    @param test_data: np array, input data.
    @param model: {W1: 1st set of weights, b1: 1st set of biases,
                   W2: 2nd set of weights, b2: 2nd set of biases}
    @return np array, output node values
    '''
    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]
    A2 = sigmoid(np.matmul(test_data, W1) + b1)
    A3 = np.matmul(A2, W2) + b2 
    return A3

housing = fetch_california_housing()
num_test = 10
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(housing.data[:-num_test, :])
y_train = housing.target[:-num_test].reshape(-1, 1)
X_test = scaler.transform(housing.data[-num_test:, :])
y_test = housing.target[-num_test:]
n_hidden_neurons = 20
learning_rate = 0.1
n_iter = 2000 
model = train(X_train, y_train, n_hidden_neurons, learning_rate, n_iter)
print(predict(X_test, model))
print(y_test)