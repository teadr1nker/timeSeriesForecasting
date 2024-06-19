import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def testForest(data):
    prices = data['Adj Close'].values

    size = len(prices)
    split = .8
    a = int(size * split)

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

    return [(y_pred_test, y_test)]
