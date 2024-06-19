import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import math

import keras
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

seed = np.random.randint(0, 99999)
print('Seed:', seed)

keras.utils.set_random_seed(seed)


scaler = MinMaxScaler(feature_range=(0,1))



def rescale(res):
    pred = [y for y in scaler.inverse_transform(res[0])]
    test = [y for y in scaler.inverse_transform(res[1])]
    return (pred, test)


def create_sequences(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        target.append(data[i+seq_length])
    return np.array(sequences), np.array(target)

def RNNCreateCompileTest(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    # scaler = MinMaxScaler(feature_range=(0,1))

    regressor = Sequential()

    # adding RNN layers and dropout regularization
    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True,
                            input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add(SimpleRNN(units = 25,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add( SimpleRNN(units = 50))

    # adding the output layer
    regressor.add(Dense(units = 1,activation='sigmoid'))

    # compiling RNN
    regressor.compile(optimizer = SGD(learning_rate=0.01,
                                    decay=1e-6,
                                    momentum=0.9,
                                    nesterov=True),
                    loss = "mean_squared_error")



    # fitting the model
    regressor.fit(X_train, y_train, epochs = 4, batch_size = 2)
    regressor.summary()

    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        #print(X)
        x = np.array([X])

        y = regressor.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)

    return rescale((y_pred, y_test))


def RNNCreateCompileTestBL(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    # scaler = MinMaxScaler(feature_range=(0,1))
    regressor = Sequential()

    # adding RNN layers and dropout regularization
    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True,
                            input_shape = (50, 3)))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add(SimpleRNN(units = 25,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add( SimpleRNN(units = 50))

    # adding the output layer
    regressor.add(Dense(units = 1,activation='sigmoid'))

    # compiling RNN
    regressor.compile(optimizer = SGD(learning_rate=0.01,
                                    decay=1e-6,
                                    momentum=0.9,
                                    nesterov=True),
                    loss = "mean_squared_error")

    X_trainBL = []

    for X in X_train:
        X = [x[0] for x in X]
        series = pd.Series(X)
        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        X = np.array([upper, X, bottom]).T[50:]
        X_trainBL.append(X)

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)
    # fitting the model
    regressor.fit(X_trainBL, y_train, epochs = 4, batch_size = 2)
    regressor.summary()

    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        x = [a[0] for a in X]
        series = pd.Series(x)
        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        # print(upper, x, bottom)
        x = np.array([upper, x, bottom]).T[50:]

        x = np.array([x])

        y = regressor.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))

def RNNCreateCompileTestRSI(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    # scaler = MinMaxScaler(feature_range=(0,1))
    regressor = Sequential()

    # adding RNN layers and dropout regularization
    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True,
                            input_shape = (50, 2)))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add(SimpleRNN(units = 25,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add( SimpleRNN(units = 50))

    # adding the output layer
    regressor.add(Dense(units = 1,activation='sigmoid'))

    # compiling RNN
    regressor.compile(optimizer = SGD(learning_rate=0.01,
                                    decay=1e-6,
                                    momentum=0.9,
                                    nesterov=True),
                    loss = "mean_squared_error")

    X_trainBL = []

    for X in X_train:
        X = [x[0] for x in X]
        series = pd.Series(X)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        X = np.array([RSI, X]).T[50:]
        X_trainBL.append(X)

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)
    # fitting the model
    regressor.fit(X_trainBL, y_train, epochs = 4, batch_size = 2)
    regressor.summary()

    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        x = [a[0] for a in X]
        series = pd.Series(x)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        x = np.array([RSI, x]).T[50:]

        x = np.array([x])

        y = regressor.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))


def RNNCreateCompileTestRSIBL(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    # scaler = MinMaxScaler(feature_range=(0,1))
    regressor = Sequential()

    # adding RNN layers and dropout regularization
    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True,
                            input_shape = (50, 4)))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add(SimpleRNN(units = 25,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add( SimpleRNN(units = 50))

    # adding the output layer
    regressor.add(Dense(units = 1,activation='sigmoid'))

    # compiling RNN
    regressor.compile(optimizer = SGD(learning_rate=0.01,
                                    decay=1e-6,
                                    momentum=0.9,
                                    nesterov=True),
                    loss = "mean_squared_error")

    X_trainBL = []

    for X in X_train:
        X = [x[0] for x in X]
        series = pd.Series(X)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        X = np.array([RSI, upper, X, bottom]).T[50:]
        X_trainBL.append(X)

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)
    # fitting the model
    regressor.fit(X_trainBL, y_train, epochs = 4, batch_size = 2)
    regressor.summary()

    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        x = [a[0] for a in X]
        series = pd.Series(x)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        x = np.array([RSI, upper, x, bottom]).T[50:]

        x = np.array([x])

        y = regressor.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))



def LSTMCreateCompileTest(X_train, X_test, y_train, y_test):

    print(X_train[0].shape)

    # scaler = MinMaxScaler(feature_range=(0,1))



    regressorLSTM = Sequential()

    #Adding LSTM layers
    regressorLSTM.add(LSTM(50,
                        return_sequences = True,
                        input_shape = (X_train.shape[1], 1)))
    regressorLSTM.add(LSTM(50,
                        return_sequences = False))
    regressorLSTM.add(Dense(25))

    #Adding the output layer
    regressorLSTM.add(Dense(1))

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

    # y_pred = regressorLSTM.predict(X_test)
    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        #print(X)
        x = np.array([X])

        y = regressorLSTM.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)
    return rescale((y_pred, y_test))


def LSTMCreateCompileTestBL(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    regressorLSTM = Sequential()

    #Adding LSTM layers
    regressorLSTM.add(LSTM(50,
                        return_sequences = True,
                        input_shape = (50, 3)))
    regressorLSTM.add(LSTM(50,
                        return_sequences = False))
    regressorLSTM.add(Dense(25))

    #Adding the output layer
    regressorLSTM.add(Dense(1))

    #Compiling the model
    regressorLSTM.compile(optimizer = 'adam',
                        loss = 'mean_squared_error',
                        metrics = ["accuracy"])



    X_trainBL = []

    for X in X_train:
        X = [x[0] for x in X]
        series = pd.Series(X)
        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        X = np.array([upper, X, bottom]).T[50:]
        X_trainBL.append(X)

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)

    #Fitting the model
    regressorLSTM.fit(X_trainBL,
                    y_train,
                    batch_size = 1,
                    epochs = 4)
    regressorLSTM.summary()

    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        x = [a[0] for a in X]
        series = pd.Series(x)
        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        # print(upper, x, bottom)
        x = np.array([upper, x, bottom]).T[50:]

        x = np.array([x])

        y = regressorLSTM.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))

def LSTMCreateCompileTestRSI(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    # scaler = MinMaxScaler(feature_range=(0,1))
    regressorLSTM = Sequential()

    #Adding LSTM layers
    regressorLSTM.add(LSTM(50,
                        return_sequences = True,
                        input_shape = (50, 2)))
    regressorLSTM.add(LSTM(50,
                        return_sequences = False))
    regressorLSTM.add(Dense(50))

    #Adding the output layer
    regressorLSTM.add(Dense(1))

    #Compiling the model
    regressorLSTM.compile(optimizer = 'adam',
                        loss = 'mean_squared_error',
                        metrics = ["accuracy"])

    X_trainBL = []

    for X in X_train:
        X = [x[0] for x in X]
        series = pd.Series(X)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        X = np.array([RSI, X]).T[50:]
        X_trainBL.append(X)

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)
    # fitting the model
    regressorLSTM.fit(X_trainBL,
                    y_train,
                    batch_size = 1,
                    epochs = 4)
    regressorLSTM.summary()
    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        x = [a[0] for a in X]
        series = pd.Series(x)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        x = np.array([RSI, x]).T[50:]

        x = np.array([x])

        y = regressorLSTM.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))


def LSTMCreateCompileTestRSIBL(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    # scaler = MinMaxScaler(feature_range=(0,1))
    regressorLSTM = Sequential()

    #Adding LSTM layers
    regressorLSTM.add(LSTM(50,
                        return_sequences = True,
                        input_shape = (50, 4)))
    regressorLSTM.add(LSTM(50,
                        return_sequences = False))
    regressorLSTM.add(Dense(50))

    #Adding the output layer
    regressorLSTM.add(Dense(1))

    #Compiling the model
    regressorLSTM.compile(optimizer = 'adam',
                        loss = 'mean_squared_error',
                        metrics = ["accuracy"])

    X_trainBL = []

    for X in X_train:
        X = [x[0] for x in X]
        series = pd.Series(X)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        X = np.array([RSI, upper, X, bottom]).T[50:]
        X_trainBL.append(X)

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)
    # fitting the model
    regressorLSTM.fit(X_trainBL, y_train, epochs = 4, batch_size = 2)
    regressorLSTM.summary()

    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        x = [a[0] for a in X]
        series = pd.Series(x)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        x = np.array([RSI, upper, x, bottom]).T[50:]

        x = np.array([x])

        y = regressorLSTM.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))


def GRUCreateCompileTest(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    # scaler = MinMaxScaler(feature_range=(0,1))



    regressorGRU = Sequential()

    # GRU layers with Dropout regularisation
    regressorGRU.add(GRU(units=50,
                        return_sequences=True,
                        input_shape=(X_train.shape[1], 1),
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
    regressorGRU.add(Dense(units=1,
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

    # y_pred = regressorGRU.predict(X_test)
    print(X_test, X_test.shape)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        #print(X)
        x = np.array([X])

        y = regressorGRU.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)
    return rescale((y_pred, y_test))


def GRUCreateCompileTestBL(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    regressorGRU = Sequential()

    # GRU layers with Dropout regularisation
    regressorGRU.add(GRU(units=50,
                        return_sequences=True,
                        input_shape=(50, 3),
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
    regressorGRU.add(Dense(units=1,
                        activation='relu'))
    # Compiling the RNN
    regressorGRU.compile(optimizer=SGD(learning_rate=0.01,
                                    decay=1e-7,
                                    momentum=0.9,
                                    nesterov=False),
                        loss='mean_squared_error')


    X_trainBL = []

    for X in X_train:
        X = [x[0] for x in X]
        series = pd.Series(X)
        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        X = np.array([upper, X, bottom]).T[50:]
        X_trainBL.append(X)

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)

    #Fitting the model
    regressorGRU.fit(X_trainBL, y_train, epochs = 4, batch_size = 2)
    regressorGRU.summary()


    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        x = [a[0] for a in X]
        series = pd.Series(x)
        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        # print(upper, x, bottom)
        x = np.array([upper, x, bottom]).T[50:]

        x = np.array([x])

        y = regressorGRU.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))

def GRUCreateCompileTestRSI(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)

    regressorGRU = Sequential()

    # GRU layers with Dropout regularisation
    regressorGRU.add(GRU(units=50,
                        return_sequences=True,
                        input_shape=(50, 2),
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
    regressorGRU.add(Dense(units=1,
                        activation='relu'))
    # Compiling the RNN
    regressorGRU.compile(optimizer=SGD(learning_rate=0.01,
                                    decay=1e-7,
                                    momentum=0.9,
                                    nesterov=False),
                        loss='mean_squared_error')


    X_trainBL = []

    for X in X_train:
        X = [x[0] for x in X]
        series = pd.Series(X)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        X = np.array([RSI, X]).T[50:]
        X_trainBL.append(X)

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)
    # fitting the model
    regressorGRU.fit(X_trainBL, y_train, epochs = 4, batch_size = 2)
    regressorGRU.summary()
    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        x = [a[0] for a in X]
        series = pd.Series(x)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        x = np.array([RSI, x]).T[50:]

        x = np.array([x])

        y = regressorGRU.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)
    return rescale((y_pred, y_test))


def GRUCreateCompileTestRSIBL(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)


    regressorGRU = Sequential()

    # GRU layers with Dropout regularisation
    regressorGRU.add(GRU(units=50,
                        return_sequences=True,
                        input_shape=(50, 4),
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
    regressorGRU.add(Dense(units=1,
                        activation='relu'))
    # Compiling the RNN
    regressorGRU.compile(optimizer=SGD(learning_rate=0.01,
                                    decay=1e-7,
                                    momentum=0.9,
                                    nesterov=False),
                        loss='mean_squared_error')


    X_trainBL = []

    for X in X_train:
        X = [x[0] for x in X]
        series = pd.Series(X)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        X = np.array([RSI, upper, X, bottom]).T[50:]
        X_trainBL.append(X)

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)
    # fitting the model
    regressorGRU.fit(X_trainBL, y_train, epochs = 4, batch_size = 2)
    regressorGRU.summary()

    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        x = [a[0] for a in X]
        series = pd.Series(x)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        x = np.array([RSI, upper, x, bottom]).T[50:]

        x = np.array([x])

        y = regressorGRU.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))


def GRUCreateCompileTestPSAR(X_train, X_test, y_train, y_test):
    print(X_train[0].shape)


    regressorGRU = Sequential()

    # GRU layers with Dropout regularisation
    regressorGRU.add(GRU(units=50,
                        return_sequences=True,
                        input_shape=(50, 4),
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
    regressorGRU.add(Dense(units=1,
                        activation='relu'))
    # Compiling the RNN
    regressorGRU.compile(optimizer=SGD(learning_rate=0.01,
                                    decay=1e-7,
                                    momentum=0.9,
                                    nesterov=False),
                        loss='mean_squared_error')


    X_trainBL = []

    for X in X_train:
        X = [x[0] for x in X]
        series = pd.Series(X)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        X = np.array([RSI, upper, X, bottom]).T[50:]
        X_trainBL.append(X)

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)
    # fitting the model
    regressorGRU.fit(X_trainBL, y_train, epochs = 4, batch_size = 2)
    regressorGRU.summary()

    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        x = [a[0] for a in X]
        series = pd.Series(x)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = (100 - (100 / (1.0 + rs)))/100

        x = np.array([RSI, upper, x, bottom]).T[50:]

        x = np.array([x])

        y = regressorGRU.predict(x)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))


def test_RNN(data):


    data = scaler.fit_transform(data.reshape(-1, 1))
    size = len(data)
    Y = data
    X = np.arange(size)
    train_size = int(0.8 * len(X))

    train_data, test_data = Y[:train_size], Y[train_size:]
    train_dataPSAR, test_dataPSAR = Y[:train_size], Y[train_size:]


    print(train_size)

    seq_length=50
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    X_trainBL, y_trainBL = create_sequences(train_data, seq_length + 50)
    X_testBL, y_testBL = create_sequences(test_data, seq_length + 50)

    return (RNNCreateCompileTest(X_train, X_test, y_train, y_test),
            LSTMCreateCompileTest(X_train, X_test, y_train, y_test),
            GRUCreateCompileTest(X_train, X_test, y_train, y_test),
            RNNCreateCompileTestBL(X_trainBL, X_testBL, y_trainBL, y_testBL),
            LSTMCreateCompileTestBL(X_trainBL, X_testBL, y_trainBL, y_testBL),
            GRUCreateCompileTestBL(X_trainBL, X_testBL, y_trainBL, y_testBL),
            RNNCreateCompileTestRSI(X_trainBL, X_testBL, y_trainBL, y_testBL),
            LSTMCreateCompileTestRSI(X_trainBL, X_testBL, y_trainBL, y_testBL),
            GRUCreateCompileTestRSI(X_trainBL, X_testBL, y_trainBL, y_testBL),
            # RNNCreateCompileTestRSIBL(X_trainBL, X_testBL, y_trainBL, y_testBL),
            # LSTMCreateCompileTestRSIBL(X_trainBL, X_testBL, y_trainBL, y_testBL),
            # GRUCreateCompileTestRSIBL(X_trainBL, X_testBL, y_trainBL, y_testBL)
            )
