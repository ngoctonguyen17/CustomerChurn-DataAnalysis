from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_ar_model(test_data, predictions):
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse}

def evaluate_arma_model(test_data, fc_series_arma):
    mse = mean_squared_error(test_data, fc_series_arma)
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse}

def evaluate_arima_model(test_data, fc_series_arima):
    mse = mean_squared_error(test_data, fc_series_arima)
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse}

def evaluate_sarima_model(test_data, fc_series_sarima):
    mse = mean_squared_error(test_data, fc_series_sarima)
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse}

def evaluate_hw_model(test_data, hw_forecast_series):
    mse = mean_squared_error(test_data, hw_forecast_series)
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse}