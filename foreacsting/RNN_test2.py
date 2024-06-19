import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import GRU, Bidirectional
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        target.append(data[i+seq_length])
    return np.array(sequences), np.array(target)

def RNNCreateCompileTest(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    scaler = MinMaxScaler(feature_range=(0,1))

    regressor = Sequential()

    # adding RNN layers and dropout regularization
    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True,
                            input_shape = X_test[0].shape))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add(SimpleRNN(units = 25,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add( SimpleRNN(units = 50))

    # adding the output layer
    regressor.add(Dense(units = len(y_train[0]),activation='sigmoid'))

    # compiling RNN
    regressor.compile(optimizer = SGD(learning_rate=0.01,
                                    decay=1e-6,
                                    momentum=0.9,
                                    nesterov=True),
                    loss = "mean_squared_error")



    # fitting the model
    regressor.fit(X_train, y_train, epochs = 4, batch_size = 2)
    regressor.summary()

    y_pred = regressor.predict(X_test)

    return (y_pred, y_test)

def LSTMCreateCompileTest(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    scaler = MinMaxScaler(feature_range=(0,1))



    regressorLSTM = Sequential()

    #Adding LSTM layers
    regressorLSTM.add(LSTM(50,
                        return_sequences = True,
                        input_shape = X_test[0].shape))
    regressorLSTM.add(LSTM(50,
                        return_sequences = False))
    regressorLSTM.add(Dense(25))

    #Adding the output layer
    regressorLSTM.add(Dense(len(y_train[0])))

    #Compiling the model
    regressorLSTM.compile(optimizer = 'adam',
                        loss = 'mean_squared_error',
                        metrics = ["accuracy"])

    #Fitting the model
    regressorLSTM.fit(X_train,
                    y_train,
                    batch_size = 1,
                    epochs = 4)
    regressorLSTM.summary()

    y_pred = regressorLSTM.predict(X_test)

    return (y_pred, y_test)

def GRUCreateCompileTest(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    scaler = MinMaxScaler(feature_range=(0,1))



    regressorGRU = Sequential()

    # GRU layers with Dropout regularisation
    regressorGRU.add(GRU(units=50,
                        return_sequences=True,
                        input_shape=X_test[0].shape,
                        activation='tanh'))
    regressorGRU.add(Dropout(0.2))

    regressorGRU.add(GRU(units=50,
                        return_sequences=True,
                        activation='tanh'))

    regressorGRU.add(GRU(units=50,
                        return_sequences=True,
                        activation='tanh'))

    regressorGRU.add(GRU(units=50,
                        activation='tanh'))

    # The output layer
    regressorGRU.add(Dense(units=len(y_train[0]),
                        activation='relu'))
    # Compiling the RNN
    regressorGRU.compile(optimizer=SGD(learning_rate=0.01,
                                    decay=1e-7,
                                    momentum=0.9,
                                    nesterov=False),
                        loss='mean_squared_error')

    # Fitting the data
    regressorGRU.fit(X_train,y_train,epochs=4,batch_size=1)
    regressorGRU.summary()


    # fitting the model
    regressorGRU.fit(X_train, y_train, epochs = 4, batch_size = 2)
    regressorGRU.summary()

    y_pred = regressorGRU.predict(X_test)

    return (y_pred, y_test)

def test_RNN(data):

    size = len(data)
    Y = data
    X = np.arange(size)
    train_size = int(0.8 * len(X))
    train_data, test_data = Y[:train_size], Y[train_size:]

    print(train_size)

    seq_length=50
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    return (RNNCreateCompileTest(X_train, X_test, y_train, y_test),
            LSTMCreateCompileTest(X_train, X_test, y_train, y_test),
            GRUCreateCompileTest(X_train, X_test, y_train, y_test))
