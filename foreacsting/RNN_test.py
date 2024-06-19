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

def testRNN(data):

    pd.set_option('display.max_rows', 4)
    pd.set_option('display.max_columns',5)



    # Setting 80 percent data for training
    training_data_len = int(len(data) * .8)

    #Splitting the dataset
    train_data = data[:training_data_len]['Adj Close']
    test_data = data[training_data_len:]['Adj Close']

    # Selecting Open Price values
    dataset_train = train_data.values
    # Reshaping 1D to 2D array
    dataset_train = np.reshape(dataset_train, (-1,1))
    dataset_train.shape



    scaler = MinMaxScaler(feature_range=(0,1))
    # scaling dataset
    scaled_train = scaler.fit_transform(dataset_train)

    # print(scaled_train[:5])


    # Selecting Open Price values
    dataset_test = test_data.values
    # Reshaping 1D to 2D array
    dataset_test = np.reshape(dataset_test, (-1,1))
    # Normalizing values between 0 and 1
    scaled_test = scaler.fit_transform(dataset_test)
    # print(*scaled_test[:5])


    X_train = []
    y_train = []

    seq = 50
    for i in range(seq, len(scaled_train)):
        X_train.append(scaled_train[i-seq:i, 0])
        y_train.append(scaled_train[i, 0])


    X_test = []
    y_test = []
    for i in range(seq, len(scaled_test)):
        X_test.append(scaled_test[i-seq:i, 0])
        y_test.append(scaled_test[i, 0])


    # The data is converted to Numpy array
    X_train, y_train = np.array(X_train), np.array(y_train)

    #Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
    y_train = np.reshape(y_train, (y_train.shape[0],1))
    # print("X_train :",X_train.shape,"y_train :",y_train.shape)

    # The data is converted to numpy array
    X_test, y_test = np.array(X_test), np.array(y_test)

    #Reshaping
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    # print("X_test :",X_test.shape,"y_test :",y_test.shape)


    # initializing the RNN
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

    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add( SimpleRNN(units = 25))

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


    #Initialising the model
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


    #Initialising the model
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


    # predictions with X_test data
    y_RNN = regressor.predict(X_test)
    y_LSTM = regressorLSTM.predict(X_test)
    y_GRU = regressorGRU.predict(X_test)

    # scaling back from 0-1 to original
    y_RNN_O = scaler.inverse_transform(y_RNN)
    y_LSTM_O = scaler.inverse_transform(y_LSTM)
    y_GRU_O = scaler.inverse_transform(y_GRU)

    y_test = scaler.inverse_transform(y_test)

    return ((y_RNN_O, y_test), (y_LSTM_O, y_test), (y_GRU_O, y_test))
