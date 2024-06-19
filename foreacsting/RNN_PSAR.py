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

seed = 30245
print('Seed:', seed)

keras.utils.set_random_seed(seed)


scaler = MinMaxScaler(feature_range=(0,1))



def psar(barsdata, iaf = 0.02, maxaf = 0.2):
    barsdata = barsdata.T
    length = len(barsdata[0])
    dates = list(np.arange(length))
    high = list(barsdata[2])
    low = list(barsdata[1])
    close = list(barsdata[0])
    psar = close[0:len(close)]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    ep = low[0]
    hp = high[0]
    lp = low[0]
    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf
        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
    return np.array([close, low, high, psar]).T



def rescale(res):
    pred = [y for y in scaler.inverse_transform(res[0])]
    test = [y for y in scaler.inverse_transform(res[1])]
    return (np.array(pred).T[0], np.array(test).T[0])


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
                            input_shape = (20, 4)))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add(SimpleRNN(units = 25,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add( SimpleRNN(units = 50))

    # adding the output layer
    regressor.add(Dense(units = 3,activation='sigmoid'))

    # compiling RNN
    regressor.compile(optimizer = SGD(learning_rate=0.01,
                                    decay=1e-6,
                                    momentum=0.9,
                                    nesterov=True),
                    loss = "mean_squared_error")

    X_trainBL = []

    for X in X_train:
        X_trainBL.append(psar(X))

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape, 'train')
    print(y_train.shape, 'train fr')
    # fitting the model
    regressor.fit(X_trainBL, y_train, epochs = 4, batch_size = 2)
    regressor.summary()

    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):

        # print(upper, x, bottom)
        x = np.array([psar(np.array(X))])

        # print(x.shape, 'testX')
        y = regressor.predict(x, verbose=0)

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
                        input_shape = (20, 4)))
    regressorLSTM.add(LSTM(50,
                        return_sequences = False))
    regressorLSTM.add(Dense(50))

    #Adding the output layer
    regressorLSTM.add(Dense(3))

    #Compiling the model
    regressorLSTM.compile(optimizer = 'adam',
                        loss = 'mean_squared_error',
                        metrics = ["accuracy"])


    X_trainBL = []

    for X in X_train:
        X_trainBL.append(psar(X))

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape, 'train')
    print(y_train.shape, 'train fr')
    # fitting the model
    regressorLSTM.summary()

    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):

        # print(upper, x, bottom)
        x = np.array([psar(np.array(X))])

        # print(x.shape, 'testX')
        y = regressorLSTM.predict(x, verbose=0)

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
                        input_shape=(20, 4),
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
    regressorGRU.add(Dense(units=3,
                        activation='relu'))
    # Compiling the RNN
    regressorGRU.compile(optimizer=SGD(learning_rate=0.01,
                                    decay=1e-7,
                                    momentum=0.9,
                                    nesterov=False),
                        loss='mean_squared_error')


    X_trainBL = []

    for X in X_train:
        X_trainBL.append(psar(X))

    # print(X_trainBL)
    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape, 'train')
    print(y_train.shape, 'train fr')
    # fitting the model
    regressorGRU.fit(X_trainBL, y_train, epochs = 4, batch_size = 2)
    regressorGRU.summary()

    # y_pred = regressor.predict(X_test)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):

        # print(upper, x, bottom)
        x = np.array([psar(np.array(X))])

        # print(x.shape, 'testX')
        y = regressorGRU.predict(x, verbose=0)

        y_pred.append(y[0])

        X = X[1:] + list(y)


    return rescale((y_pred, y_test))



def testRNNPSAR(data):
    data = scaler.fit_transform(data)
    size = len(data)
    Y = data
    X = np.arange(size)
    train_size = int(0.8 * len(X))

    train_data, test_data = Y[:train_size], Y[train_size:]
    train_dataPSAR, test_dataPSAR = Y[:train_size], Y[train_size:]
    seq_length=20
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    return GRUCreateCompileTest(X_train, X_test, y_train, y_test)

