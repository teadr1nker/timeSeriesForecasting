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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

seed = 30245
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


def testForest(prices):

    size = len(prices)
    split = .8
    a = int(size * split) + 100

    train = prices[:a]
    test = prices[a:]

    t = np.arange(size)
    y = prices

    t_train = t[:a].reshape(-1,1)
    t_test = t[a:].reshape(-1,1)

    n_lags = 2

    y_train = y[:a]
    X_train_shift = pd.concat([pd.DataFrame(y_train).shift(t) for t in range(1,n_lags)],axis=1).diff().values[n_lags:,:]
    y_train_shift = np.diff(y_train)[n_lags-1:]
    y_test = y[a:]

    # print(X_train_shift, y_train_shift)
    # quit()

    forest = RandomForestRegressor(n_estimators=128)
    forest.random_state = 0
    forest.fit(X_train_shift, y_train_shift)

    y_pred_train = forest.predict(X_train_shift).reshape(-1)

    Xt = np.concatenate([X_train_shift[-1,1:].reshape(1,-1),np.array(y_train_shift[-1]).reshape(1,1)],1)
    predictions_test = []

    for t in range(len(y_test)):
        pred = forest.predict(Xt)
        predictions_test.append(pred[0])
        Xt = np.concatenate([np.array(pred).reshape(1,1),Xt[-1,1:].reshape(1,-1)],1)

    y_pred_test = np.array(predictions_test)

    y_pred_train = y_train[n_lags-2]+np.cumsum(y_pred_train)
    y_pred_test = y_train[-1]+np.cumsum(y_pred_test)

    return (y_pred_test, y_test)


def GRUCreateCompileTestRSI(X_train, X_test, y_train, y_test, l):
    print(X_train[0].shape)

    regressorGRU = Sequential()

    # GRU layers with Dropout regularisation
    regressorGRU.add(GRU(units=50,
                        return_sequences=True,
                        input_shape=(l, 2),
                        activation='tanh'))
    regressorGRU.add(Dropout(0.1))

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

        y = regressorGRU.predict(x, verbose=0)

        y_pred.append(y[0])

        X = X[1:] + list(y)
    return rescale((y_pred, y_test))


def LSTMCreateCompileTestRSI(X_train, X_test, y_train, y_test, l):
    print(X_train[0].shape)

    # scaler = MinMaxScaler(feature_range=(0,1))
    regressorLSTM = Sequential()

    #Adding LSTM layers
    regressorLSTM.add(LSTM(50,
                        return_sequences = True,
                        input_shape = (l, 2)))
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

        y = regressorLSTM.predict(x, verbose=0)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))


def RNNCreateCompileTestBL(X_train, X_test, y_train, y_test, l):
    print(X_train[0].shape)

    # scaler = MinMaxScaler(feature_range=(0,1))
    regressor = Sequential()

    # adding RNN layers and dropout regularization
    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True,
                            input_shape = (l, 3)))
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

        y = regressor.predict(x, verbose=0)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))


def LSTMCreateCompileTestBL(X_train, X_test, y_train, y_test, l):
    print(X_train[0].shape)

    regressorLSTM = Sequential()

    #Adding LSTM layers
    regressorLSTM.add(LSTM(50,
                        return_sequences = True,
                        input_shape = (l, 3)))
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
                    epochs = 6)
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

        y = regressorLSTM.predict(x, verbose=0)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))


def GRUCreateCompileTestBL(X_train, X_test, y_train, y_test, l):
    print(X_train[0].shape)

    regressorGRU = Sequential()

    # GRU layers with Dropout regularisation
    regressorGRU.add(GRU(units=50,
                        return_sequences=True,
                        input_shape=(l, 3),
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

        y = regressorGRU.predict(x, verbose=0)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))


def testBest(data, l = 50):
    prices = data
    data = scaler.fit_transform(data.reshape(-1, 1))
    size = len(data)
    Y = data
    X = np.arange(size)
    train_size = int(0.8 * len(X))
    train_data, test_data = Y[:train_size], Y[train_size:]

    print(train_size)

    seq_length = l

    X_trainBL, y_trainBL = create_sequences(train_data, l + 50)
    X_testBL, y_testBL = create_sequences(test_data, l + 50)
    return (scaler.inverse_transform(y_trainBL),
            testForest(prices),
            RNNCreateCompileTestBL(X_trainBL, X_testBL, y_trainBL, y_testBL, l),
            LSTMCreateCompileTestBL(X_trainBL, X_testBL, y_trainBL, y_testBL, l),
            GRUCreateCompileTestBL(X_trainBL, X_testBL, y_trainBL, y_testBL, l),
            # LSTMCreateCompileTestRSI(X_trainBL, X_testBL, y_trainBL, y_testBL),
            GRUCreateCompileTestRSI(X_trainBL, X_testBL, y_trainBL, y_testBL, l))
