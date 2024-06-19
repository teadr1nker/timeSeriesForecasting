import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout
from keras.optimizers import SGD

def create_sequences(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        target.append(data[i+seq_length])
    return np.array(sequences), np.array(target)

def CNNCreateCompileTest(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=2)
    y_pred = model.predict(X_test)

    # Plot the original and predicted time series
    # plt.figure(figsize=(12, 6))
    # plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test,
    # label='True')
    # plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred,
    # label='Predicted')
    # plt.legend()
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('Time Series Forecasting with CNN')
    # plt.show()
    return (y_pred, y_test)

def test_CNN(data):

    size = len(data)
    Y = data['Adj Close'].values
    X = np.arange(size)

    train_size = int(0.8 * len(X))
    train_data, test_data = Y[:train_size], Y[train_size:]

    seq_length=50
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)



    return (CNNCreateCompileTest(X_train, X_test, y_train, y_test), y_train)
