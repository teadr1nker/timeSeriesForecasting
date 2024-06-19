import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout
from keras.optimizers import SGD
import keras

# seed = np.random.randint(0, 99999)
seed = 30245
print('Seed:', seed)

keras.utils.set_random_seed(seed)

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
    # print(X_test.shape)
    # X = np.array([X_test[0]])
    # print(X.shape)
    # y_pred = model.predict(X)

    y_pred = []
    X = list(X_test[0])

    for i in range(len(y_test)):
        # print(X, len(X))
        x = np.array([X])

        y = model.predict(x)

        y_pred.append(y[0][0])

        X = X[1:] + list(y[0])

    return (y_pred, y_test)

def CNNCreateCompileTestBL(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(50, 3)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    X_trainBL = []

    for x in X_train:
        series = pd.Series(x)
        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2
        X = np.array([upper, x, bottom])
        # print(X.T)
        # quit(1)
        X_trainBL.append(X.T[50:])

    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)
    # Train the model
    model.fit(X_trainBL, y_train, epochs=10, batch_size=16, verbose=2)

    y_pred = []
    X = list(X_test[0])

    # print(len(y_test), 'test')

    for i in range(len(y_test)):

        series = pd.Series(X)
        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2
        x = np.array([np.array([upper, X, bottom]).T[50:]])

        y = model.predict(x)

        y_pred.append(y[0][0])

        X = X[1:] + list(y[0])

    return (y_pred, y_test)

def CNNCreateCompileTestRSIBL(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(50, 4)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    X_trainBL = []

    for x in X_train:
        series = pd.Series(x)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        upper  = series.rolling(window=50).mean() + series.rolling(window=50).std() * 2
        bottom = series.rolling(window=50).mean() - series.rolling(window=50).std() * 2

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = 100 - (100 / (1.0 + rs))

        X = np.array([RSI, upper, x, bottom])
        # print(X.T)
        # quit(1)
        X_trainBL.append(X.T[50:])

    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)
    # Train the model
    model.fit(X_trainBL, y_train, epochs=10, batch_size=16, verbose=2)

    y_pred = []
    X = list(X_test[0])

    # print(len(y_test), 'test')

    for i in range(len(y_test)):
        series = pd.Series(X)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = 100 - (100 / (1.0 + rs))

        x = np.array([np.array([RSI, upper, X, bottom]).T[50:]])

        # print(X, len(X))
        # print(x.shape)

        #print(x, x.shape)

        # x = np.array([X])

        y = model.predict(x)

        # print(y)

        y_pred.append(y[0][0])

        X = X[1:] + list(y[0])

    return (y_pred, y_test)



def CNNCreateCompileTestRSI(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(50, 2)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    X_trainBL = []

    for x in X_train:
        series = pd.Series(x)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = 100 - (100 / (1.0 + rs))

        X = np.array([RSI, x])
        # print(X.T)
        # quit(1)
        X_trainBL.append(X.T[50:])

    X_trainBL = np.array(X_trainBL)
    print(X_trainBL.shape)
    # Train the model
    model.fit(X_trainBL, y_train, epochs=10, batch_size=16, verbose=2)

    y_pred = []
    X = list(X_test[0])

    # print(len(y_test), 'test')

    for i in range(len(y_test)):
        series = pd.Series(X)
        profit  = series.diff(1)
        gain = profit.clip(lower = 0)
        loss = profit.clip(upper = 0).abs()

        rs = gain.rolling(window=50).mean() / loss.rolling(window=50).mean()
        RSI = 100 - (100 / (1.0 + rs))

        x = np.array([np.array([RSI, X]).T[50:]])

        # print(X, len(X))
        # print(x.shape)

        #print(x, x.shape)

        # x = np.array([X])

        y = model.predict(x)

        # print(y)

        y_pred.append(y[0][0])

        X = X[1:] + list(y[0])

    return (y_pred, y_test)


def test_CNN(data):

    size = len(data)
    Y = data['Adj Close'].values
    # X = np.arange(size)

    train_size = int(0.8 * len(Y))
    train_data, test_data = Y[:train_size], Y[train_size:]

    seq_length=50
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    X_trainBL, y_trainBL = create_sequences(train_data, seq_length + 50)
    X_testBL, y_testBL = create_sequences(test_data, seq_length + 50)


    return (CNNCreateCompileTest(X_train, X_test, y_train, y_test),
            CNNCreateCompileTestBL(X_trainBL, X_testBL, y_trainBL, y_testBL),
            CNNCreateCompileTestRSI(X_trainBL, X_testBL, y_trainBL, y_testBL),
            CNNCreateCompileTestRSIBL(X_trainBL, X_testBL, y_trainBL, y_testBL))
