import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Vẽ biểu đồ cho các kiểm tra trước khi huấn luyện mô hình
def plot_close_prices(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], color='blue')
    plt.xlabel('Date')
    plt.ylabel('Close prices from 2014 to 2024')
    plt.title('Closing Prices Over Time')
    plt.grid(True)
    plt.show()

def plot_close_price_histogram(df, bins=30):
    plt.figure(figsize=(10, 6))
    df['Close'].hist(bins=bins)
    plt.xlabel('Close Price')
    plt.ylabel('Frequency')
    plt.title('Histogram of Close Prices')
    plt.grid(True)
    plt.show()

def plot_log_close_prices(df_close_log):
    plt.figure(figsize=(12, 6))
    plt.plot(df_close_log, color='blue')
    plt.xlabel("Date")
    plt.ylabel("Log of Close Prices")
    plt.title("Close Prices Over Time (Log-Transformed)")
    plt.grid(True)
    plt.show()

def correlation_plot(data):
    pd.plotting.lag_plot(data)
    plt.title("Auto Correlation")
    plt.show()

def pacf(data):
    plot_pacf(data)
    plt.show()

def acf(data):
    plot_acf(data)
    plt.show()

# Vẽ biểu đồ cho Simple Split
def plot_ar_pred(train_data, test_data, pred_ar):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label="Train data")
    plt.plot(test_data, color="orange", label="Actual price")
    plt.plot(pred_ar, color="red", label="Predicted price")
    plt.fill_between(test_data.index, pred_ar, test_data, color="blue", alpha=.2)
    plt.title("Close price prediction (AR)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc='upper left', fontsize=16)
    plt.show()

def plot_arma_pred(train_data, test_data, fc_series_arma, lower_series_arma, upper_series_arma):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label="Train data")
    plt.plot(test_data, color="orange", label="Actual price")
    plt.plot(fc_series_arma, color="red", label="Predicted price (ARMA)")
    plt.fill_between(lower_series_arma.index, lower_series_arma, upper_series_arma, color="blue", alpha=.2)
    plt.title("Close price prediction (ARMA)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc='upper left', fontsize=16)
    plt.show()

def plot_arima_pred(train_data, test_data, fc_series, lower_series, upper_series):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label="Train data")
    plt.plot(test_data, color="orange", label="Actual price")
    plt.plot(fc_series, color="red", label="Predicted price")
    plt.fill_between(lower_series.index, lower_series, upper_series, color="blue", alpha=.2)
    plt.title("Close price prediction (ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc='upper left', fontsize=16)
    plt.show()

def plot_sarima_pred(train_data, test_data, fc_series_sarima, lower_series_sarima, upper_series_sarima):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label="Train data")
    plt.plot(test_data, color="orange", label="Actual price")
    plt.plot(fc_series_sarima, color="red", label="Predicted price (SARIMA)")
    plt.fill_between(lower_series_sarima.index, lower_series_sarima, upper_series_sarima, color="blue", alpha=.2)
    plt.title("Close price prediction (SARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc='upper left', fontsize=16)
    plt.show()

def plot_hw_pred(train_data, test_data, hw_forecast_series):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label="Train data")
    plt.plot(test_data, color="orange", label="Actual price")
    plt.plot(hw_forecast_series, color="red", label="Predicted price (Holt-Winters)")
    plt.title("Close price prediction (Holt-Winters)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc='upper left', fontsize=16)
    plt.show()

# Vẽ biểu đồ cho Rolling Window
def plot_rolling_ar_pred(df_close_log, train_data, test_data, pred_ar, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(df_close_log, color="black", label="Actual Data")
    plt.plot(train_data.index, train_data, color="blue", label="Train Data")
    plt.plot(test_data.index, test_data, color="orange", label="Test Data")
    plt.plot(test_data.index, pred_ar, color="red", label="AR Prediction")
    plt.title(f"AR Model Rolling Window {fold + 1}")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá")
    plt.legend(loc="upper left")
    plt.show()

def plot_rolling_arma_pred(df_close_log, train_data, test_data, fc_series_arma, lower_series_arma, upper_series_arma, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(df_close_log, color="black", label="Actual Data")
    plt.plot(train_data.index, train_data, color="blue", label="Train Data")
    plt.plot(test_data.index, test_data, color="orange", label="Test Data")
    plt.plot(test_data.index, fc_series_arma, color="red", label="ARMA Prediction")
    plt.fill_between(test_data.index, lower_series_arma, upper_series_arma, color="blue", alpha=0.3)
    plt.title(f"ARMA Model Rolling Window {fold + 1}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.show()

def plot_rolling_arima_pred(df_close_log, train_data, test_data, fc_series_arima, lower_series_arima, upper_series_arima, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(df_close_log, color="black", label="Actual Data")
    plt.plot(train_data.index, train_data, color="blue", label="Train Data")
    plt.plot(test_data.index, test_data, color="orange", label="Test Data")
    plt.plot(test_data.index, fc_series_arima, color="red", label="ARIMA Prediction")
    plt.fill_between(test_data.index, lower_series_arima, upper_series_arima, color="blue", alpha=0.3)
    plt.title(f"ARIMA Model Rolling Window {fold + 1}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.show()

def plot_rolling_sarima_pred(df_close_log, train_data, test_data, fc_series_sarima, lower_series_sarima, upper_series_sarima, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(df_close_log, color="black", label="Actual Data")
    plt.plot(train_data.index, train_data, color="blue", label="Train Data")
    plt.plot(test_data.index, test_data, color="orange", label="Test Data")
    plt.plot(test_data.index, fc_series_sarima, color="red", label="SARIMA Prediction")
    plt.fill_between(test_data.index, lower_series_sarima, upper_series_sarima, color="blue", alpha=0.3)
    plt.title(f"SARIMA Model Rolling Window {fold + 1}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.show()

def plot_rolling_hw_pred(df_close_log, train_data, test_data, hw_forecast_series, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(df_close_log, color="black", label="Actual Data")
    plt.plot(train_data.index, train_data, color="blue", label="Train Data")
    plt.plot(test_data.index, test_data, color="orange", label="Test Data")
    plt.plot(test_data.index, hw_forecast_series, color="red", label="Holt-Winters Prediction")
    plt.title(f"Holt-Winters Model Rolling Window {fold + 1}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.show()

# Vẽ biểu đồ cho Expanding Window
def plot_expand_ar_pred(df_close_log, train_data, test_data, pred_ar, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(df_close_log, color="black", label="Actual Data")
    plt.plot(train_data.index, train_data, color="blue", label="Train Data")
    plt.plot(test_data.index, test_data, color="orange", label="Test Data")
    plt.plot(test_data.index, pred_ar, color="red", label="AR Prediction")
    plt.title(f"AR Model Expanding Window {fold + 1}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.show()

def plot_expand_arma_pred(df_close_log, train_data, test_data, fc_series_arma, lower_series_arma, upper_series_arma, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(df_close_log, color="black", label="Actual Data")
    plt.plot(train_data.index, train_data, color="blue", label="Train Data")
    plt.plot(test_data.index, test_data, color="orange", label="Test Data")
    plt.plot(test_data.index, fc_series_arma, color="red", label="ARMA Prediction")
    plt.fill_between(test_data.index, lower_series_arma, upper_series_arma, color="blue", alpha=0.3)
    plt.title(f"ARMA Model Expanding Window {fold + 1}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.show()

def plot_expand_arima_pred(df_close_log, train_data, test_data, fc_series_arima, lower_series_arima, upper_series_arima, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(df_close_log, color="black", label="Actual Data")
    plt.plot(train_data.index, train_data, color="blue", label="Train Data")
    plt.plot(test_data.index, test_data, color="orange", label="Test Data")
    plt.plot(test_data.index, fc_series_arima, color="red", label="ARIMA Prediction")
    plt.fill_between(test_data.index, lower_series_arima, upper_series_arima, color="blue", alpha=0.3)
    plt.title(f"ARIMA Model Expanding Window {fold + 1}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.show()

def plot_expand_sarima_pred(df_close_log, train_data, test_data, fc_series_sarima, lower_series_sarima, upper_series_sarima, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(df_close_log, color="black", label="Actual Data")
    plt.plot(train_data.index, train_data, color="blue", label="Train Data")
    plt.plot(test_data.index, test_data, color="orange", label="Test Data")
    plt.plot(test_data.index, fc_series_sarima, color="red", label="SARIMA Prediction")
    plt.fill_between(test_data.index, lower_series_sarima, upper_series_sarima, color="blue", alpha=0.3)
    plt.title(f"SARIMA Model Expanding Window {fold + 1}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.show()

def plot_expand_hw_pred(df_close_log, train_data, test_data, hw_forecast_series, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(df_close_log, color="black", label="Actual Data")
    plt.plot(train_data.index, train_data, color="blue", label="Train Data")
    plt.plot(test_data.index, test_data, color="orange", label="Test Data")
    plt.plot(test_data.index, hw_forecast_series, color="red", label="Holt-Winters Prediction")
    plt.title(f"Holt-Winters Model Expanding Window {fold + 1}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.show()

def plot_data_rolling_splits(df, window_size, test_size, step_size, n_splits=5):
    plt.figure(figsize=(12, 6))
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i in range(n_splits):
        train_start = i * step_size
        train_end = train_start + window_size
        test_end = train_end + test_size

        if test_end > len(df):
            break

        train_data = df[train_start:train_end]
        test_data = df[train_end:test_end]
        
        plt.plot(train_data.index, train_data.values, label=f'Window {i + 1} - Train', color=colors[i % len(colors)], linestyle='--')
        plt.plot(test_data.index, test_data.values, label=f'Window {i + 1} - Test', color=colors[i % len(colors)], linestyle='-')
    
    plt.title('Data Split Visualization for Rolling Windows')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


