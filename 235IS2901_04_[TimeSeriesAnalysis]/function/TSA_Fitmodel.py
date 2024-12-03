import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from function.TSA_Function import *

# AR
def fit_ar_model(train_data, max_lag, criterion='aic'):
    p = find_optimal_p(train_data, 5)
    model_ar = AutoReg(train_data, lags=p).fit()
    print(model_ar.summary())
    
    return model_ar

def predict_ar_model(model_ar, train_data, test_data):
    pred_ar = model_ar.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)
    pred_ar.index = test_data.index
    
    return pred_ar

# ARMA
def fit_arma_model(train_data, p, q):
    model_arma = ARIMA(train_data, order=(p, 0, q)).fit()
    print(model_arma.summary())
    
    return model_arma

def predict_arma_model(fitted_arma, test_data):
    fc_arma = fitted_arma.get_forecast(len(test_data))
    fc_series_arma = fc_arma.predicted_mean
    fc_series_arma.index = test_data.index
    conf_arma = fc_arma.conf_int(alpha=0.05)
    lower_series_arma = conf_arma.iloc[:, 0]
    lower_series_arma.index = test_data.index
    upper_series_arma = conf_arma.iloc[:, 1]
    upper_series_arma.index = test_data.index
    
    return fc_series_arma, lower_series_arma, upper_series_arma

# ARIMA
def fit_arima_model(train_data):
    stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True, error_action='ignore')
    order = stepwise_fit.order
    stepwise_fit.plot_diagnostics(figsize=(15, 8))
    plt.show()
    
    model = ARIMA(train_data, order=order, trend='t')
    fitted_arima = model.fit()
    print(fitted_arima.summary())
    
    return fitted_arima, order

def forecast_arima_model(fitted_arima, test_data):
    fc = fitted_arima.get_forecast(len(test_data))
    fc_series = fc.predicted_mean
    fc_series.index = test_data.index
    conf = fc.conf_int(alpha=0.05)
    lower_series = conf['lower Close']
    lower_series.index = test_data.index
    upper_series = conf['upper Close']
    upper_series.index = test_data.index
    
    return fc_series, lower_series, upper_series

# SARIMA
def fit_sarima_model(train_data, seasonal_period=12):
    stepwise_fit_sarima = auto_arima(train_data, seasonal=True, m=seasonal_period, trace=True, suppress_warnings=True)
    seasonal_order = stepwise_fit_sarima.seasonal_order
    order_sarima = stepwise_fit_sarima.order
    stepwise_fit_sarima.plot_diagnostics(figsize=(15, 8))
    plt.show()

    model_sarima = SARIMAX(train_data, order=order_sarima, seasonal_order=seasonal_order, trend="t")
    fitted_sarima = model_sarima.fit()
    print(fitted_sarima.summary())
    
    return fitted_sarima

def predict_sarima_model(fitted_sarima, test_data):
    fc_sarima = fitted_sarima.get_forecast(steps=len(test_data))
    fc_series_sarima = fc_sarima.predicted_mean
    fc_series_sarima.index = test_data.index
    conf_sarima = fc_sarima.conf_int(alpha=0.05)
    lower_series_sarima = conf_sarima.iloc[:, 0]
    lower_series_sarima.index = test_data.index
    upper_series_sarima = conf_sarima.iloc[:, 1]
    upper_series_sarima.index = test_data.index

    return fc_series_sarima, lower_series_sarima, upper_series_sarima

# HOLT-WINTERS
def fit_hw_model(train_data, trend='add', seasonal=None, seasonal_periods=None):
    model_hw = ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fitted_hw = model_hw.fit()
    print(fitted_hw.summary())
    
    return fitted_hw

def predict_hw_model(fitted_hw, test_data):
    fc_hw = fitted_hw.forecast(steps=len(test_data))
    fc_series_hw = pd.Series(fc_hw.values, index=test_data.index)    
    return fc_series_hw
