#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import  HistGradientBoostingRegressor


df = pd.read_excel("data/data.ods", engine="odf", sheet_name='forecasting')
#print(df)


data = df.drop(columns='Year/Month')
items = list(data.columns)
print(items)
forecaster_ms = ForecasterAutoregMultiSeries(
                    regressor          = HistGradientBoostingRegressor(random_state=123),
                    lags               = 5,
                    transformer_series = StandardScaler(),
                )

multi_series_mae, predictions_ms = backtesting_forecaster_multiseries(
                                       forecaster         = forecaster_ms,
                                       series             = data,
                                       levels             = items,
                                       steps              = 7,
                                       metric             = 'mean_absolute_error',
                                       initial_train_size = 30,
                                       refit              = False,
                                       fixed_train_size   = False,
                                       verbose            = False
                                   )

print(multi_series_mae.head(3))

print(predictions_ms.head(3))

forecaster_ms.fit(series=data)
Y = forecaster_ms.predict(6, levels=items)
print(Y)
Y.to_csv('predictions.csv')
