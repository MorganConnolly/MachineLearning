import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
tf.random.set_seed(42)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorboard.plugins.hparams import api as hp
from keras.layers import Dense
from keras import Sequential
import matplotlib.pyplot as plt
"""
Trains a neural network on historical index fund data to predict the
closing price on a given day.

Utilises TensorFlow's Keras API to determine the optimal hyperparameters:
the number of hidden nodes, the number of iterations (number of times the
weights and biases are updated) and the learning rate (how fast they
change). Trains and tests the model using these hyperparameters and plots
the results using Matplotlib.

I completed this project with guidance from the following book:
https://www.amazon.co.uk/Python-Machine-Learning-Example-real-world-dp-1835085628/dp/1835085628/ref=dp_ob_title_bk
"""

def add_original_features(df, df_new):
    df_new["open"] = df["Open"]
    df_new["open_1"] = df["Open"].shift(1)
    df_new["close_1"] = df["Close"].shift(1)
    df_new["high_1"] = df["High"].shift(1)
    df_new["low_1"] = df["Low"].shift(1)
    df_new["volume_1"] = df["Volume"].shift(1)

def add_avg_price(df, df_new):
    df_new["avg_price_5"] = df["Close"].rolling(5).mean().shift(1)
    df_new["avg_price_30"] = df["Close"].rolling(21).mean().shift(1)
    df_new["avg_price_365"] = df["Close"].rolling(252).mean(1).shift(1)
    df_new["ratio_avg_price_5_30"] = df_new["avg_price_5"] / df_new["avg_price_30"]
    df_new["ratio_avg_price_5_365"] = df_new["avg_price_5"] / df_new["avg_price_365"]
    df_new["ratio_avg_price_30_365"] = df_new["avg_price_30"] / df_new["avg_price_365"]

def add_avg_volume(df, df_new):
    df_new["avg_volume_5"] = df["Volume"].rolling(5).mean().shift(1)
    df_new["avg_volume_30"] = df["Volume"].rolling(21).mean().shift(1)
    df_new["avg_volume_365"] = df["Volume"].rolling(252).mean().shift(1)
    df_new["avg_volume_365"] = df["Volume"].rolling(252).mean().shift(1)
    df_new["ratio_avg_volume_5_30"] = df_new["avg_volume_5"] / df_new["avg_volume_30"]
    df_new["ratio_avg_volume_5_365"] = df_new["avg_volume_5"] / df_new["avg_volume_365"]
    df_new["ratio_avg_volume_30_365"] = df_new["avg_volume_30"] / df_new["avg_volume_365"]

def add_std_price(df, df_new):
    df_new["std_price_5"] = df["Close"].rolling(5).std().shift(1)
    df_new["std_price_30"] = df["Close"].rolling(21).std().shift(1)
    df_new["std_price_365"] = df["Close"].rolling(252).std().shift(1)
    df_new["ratio_std_price_5_30"] = df_new["std_price_5"] / df_new["std_price_30"]
    df_new["ratio_std_price_5_365"] = df_new["std_price_5"] / df_new["std_price_365"]
    df_new["ratio_std_price_30_365"] = df_new["std_price_30"] / df_new["std_price_365"]

def add_std_volume(df, df_new):
    df_new["std_volume_5"] = df["Volume"].rolling(5).mean().shift(1)
    df_new["std_volume_30"] = df["Volume"].rolling(21).mean().shift(1)
    df_new["std_volume_365"] = df["Volume"].rolling(252).mean().shift(1)
    df_new["std_volume_365"] = df["Volume"].rolling(252).mean().shift(1)
    df_new["ratio_std_volume_5_30"] = df_new["std_volume_5"] / df_new["std_volume_30"]
    df_new["ratio_std_volume_5_365"] = df_new["std_volume_5"] / df_new["std_volume_365"]
    df_new["ratio_std_volume_30_365"] = df_new["std_volume_30"] / df_new["std_volume_365"]

def add_return_feature(df, df_new):
    df_new["return_1"] = ((df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)).shift(1)
    df_new["return_5"] = ((df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5)).shift(1)
    df_new["return_30"] = ((df["Close"] - df["Close"].shift(21)) / df["Close"].shift(21)).shift(1)
    df_new["return_365"] = ((df["Close"] - df["Close"].shift(252)) / df["Close"].shift(252)).shift(1)
    df_new["moving_avg_5"] = df_new["return_1"].rolling(5).mean().shift(1)
    df_new["moving_avg_30"] = df_new["return_1"].rolling(21).mean().shift(1)
    df_new["moving_avg_365"] = df_new["return_1"].rolling(252).mean().shift(1)

def generate_features(df):
    """
    Generate features for a stock/index based on historical
    price and performance.
    @param df: dataframe with columns "Open", "Close", "High", "Low", "Volume", "Adjusted Close"
    @return: datframe, data set with new features
    """
    df_new = pd.DataFrame()
    # Generate 6 original features.
    add_original_features(df, df_new)
    # Generate 31 additional features.
    add_avg_price(df, df_new)
    add_avg_volume(df, df_new)
    add_std_price(df, df_new)
    add_std_volume(df, df_new)
    add_return_feature(df, df_new)
    # Generate the target.
    df_new["Close"] = df["Close"]
    df_new = df_new.dropna(axis = 0)
    return df_new

# Load data & generate additional features.
data_raw = pd.read_csv("Stock Dataset.csv", index_col = "Date")
data = generate_features(data_raw)

# Initialise data.
start_train = "1971-02-05"
end_train = "2018-12-31"
start_test = "2019-01-01"
end_test = "2019-12-31"
data_train = data.loc[start_train:end_train]
X_train = data_train.drop("Close", axis = 1).values
y_train = data_train["Close"].values
data_test = data.loc[start_test:end_test]
X_test = data_test.drop("Close", axis = 1).values
y_test = data_test["Close"].values

# Normalise features to ease comparability by removing the mean and rescaling to unit variance.
# This should be done wherever gradient decent is used.
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

# # Fine-tuning the hyperparameters.
# hp_hidden = hp.HParam("hidden_size", hp.Discrete([64, 32, 16]))
# hp_epochs = hp.HParam("epochs", hp.Discrete([300, 1000]))
# hp_learning_rate = hp.HParam("learning_rate", hp.RealInterval(0.01, 0.4))

# def train_test_model(hparams, logdir):
#     """
#     Fit a neural network with a set of hyperparameters and evalute performance.
#     """
#     model = Sequential([
#         Dense(units = hparams[hp_hidden],
#               activation = "relu"),
#         Dense(units = 1)
#     ])
#     model.compile(loss = "mean_squared_error",
#                   optimizer = tf.keras.optimizers.Adam(
#                               hparams[hp_learning_rate]),
#                   metrics = ["mean_squared_error"])
#     model.fit(X_scaled_train, y_train,
#               validation_data = (X_scaled_test, y_test),
#               epochs = hparams[hp_epochs], verbose = False,
#               callbacks = [                                 # Provides visualisation for the model
#                   tf.keras.callbacks.TensorBoard(logdir),   # graph and metrics during training
#                   hp.KerasCallback(logdir, hparams),        # and validation.
#                   tf.keras.callbacks.EarlyStopping(
#                       monitor = "val_loss", min_delta = 0,
#                       patience = 200, verbose = 0,
#                       mode = "auto",
#                   )
#               ],
#               )
#
#     # Testing the model.
#     _, mse = model.evaluate(X_scaled_test, y_test)
#     pred = model.predict(X_scaled_test)
#     r2 = r2_score(y_test, pred)
#     return mse, r2

# def run(hparams, logdir):
#     """
#     Carries out training using hyperparameters and writes a summary of the
#     MSE and R^2 metrics to a folder.
#     """
#     with tf.summary.create_file_writer(logdir).as_default():
#         hp.hparams_config(
#             hparams = [hp_hidden, hp_epochs,
#                        hp_learning_rate],
#             metrics = [hp.Metric("mean_squared_error",
#                                  display_name = "mse"),
#                        hp.Metric("r2", display_name = "r2")],
#         )
#         mse, r2 = train_test_model(hparams, logdir)
#         tf.summary.scalar("mean_squared_error", mse,
#                           step = 1)
#         tf.summary.scalar("r2", r2, step = 1)

# # Train the model with different hyperparameters in a gridsearch manner.
# session_num = 1
# for hidden in hp_hidden.domain.values:
#     for epochs in hp_epochs.domain.values:
#         for learning_rate in tf.linspace(hp_learning_rate.domain.min_value, 
#                                          hp_learning_rate.domain.max_value, 5):
#             hparams = {
#                 hp_hidden: hidden,
#                 hp_epochs: epochs,
#                 hp_learning_rate:
#                     float("%.2f"%float(learning_rate))
#             }
#             run_name = "run-%d" % session_num 
#             print("--- Starting trail: %s" % run_name)
#             print({h.name: hparams[h] for h in hparams})
#             run(hparams, "logs/hparam/tuning/" + run_name)
#             session_num += 1

# # Create a neural network with two hidden layers and activation function ReLU.
# # Hyperparameters chosen randomly.
# model = Sequential([Dense(units = 32, activation = "relu"),
#                     Dense(units = 1)
# ])
# # Uses the Adam optimizer rather than stochastic gradient decent.
# model.compile(loss = "mean_squared_error",
#               optimizer = tf.keras.optimizers.Adam(0.01))
# model.fit(X_scaled_train, y_train, epochs = 100, verbose = True) # Train 100 iterations.

# predictions = model.predict(X_scaled_test)[:, 0] # Test
# print(predictions[:5])
# print(y_test[:5])

# # Calculate stats to indicate how effective the regression is.
# print(f"MSE: {mean_squared_error(y_test, predictions):.3f}")
# print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
# print(f"R^2: {r2_score(y_test, predictions):.3f}")

# Using optimal hyperparameters.
model = Sequential([Dense(units = 16, activation = "relu"),
                    Dense(units = 1)
])
# Uses the Adam optimizer rather than stochastic gradient decent.
model.compile(loss = "mean_squared_error",
              optimizer = tf.keras.optimizers.Adam(0.21))
model.fit(X_scaled_train, y_train, epochs = 100, verbose = True) # 1000 iterations.
predictions = model.predict(X_scaled_test)[:, 0] # Test

# Model predictions against the real values.
plt.plot(data_test.index, y_test, c = "k")
plt.plot(data_test.index, predictions, c = "b")
# plt.plot(data_test.index, predictions, c = "r")
# plt.plot(data_test.index, predictions, c = "g")
plt.xticks(range(0, 252, 10), rotation = 60)
plt.xlabel("Date")
plt.ylabel("Close price")
plt.legend(["Truth", "Neural network prediction"])
plt.show()