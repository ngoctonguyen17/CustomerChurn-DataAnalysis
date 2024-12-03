# %% 
import pandas as pd
import numpy as np
import seaborn as sns
from function.TSA_Fitmodel import *
from function.TSA_Evaluate import *
from function.TSA_Plot import *
from function.TSA_Function import *
import warnings
warnings.filterwarnings("ignore")

# %% 
configuration()
df = pd.read_csv("./data/AppleStock.csv", index_col="Date", parse_dates=True)
df.head()

# %%
print("DataFrame Information:")
df.info()
print("------"*10)
print("\nMissing Values in Each Column:")
print(df.isnull().sum())
print("------"*10)
print("\nNumber of Duplicate Rows:", df.duplicated().sum())

# %% Các biểu đồ 
plot_close_price_histogram(df, bins=30) #Biểu đồ histogram
plot_close_prices(df) #Biểu đồ giá đóng cửa theo thời gian

# %% 
df_close_log = np.log(df['Close'])
plot_log_close_prices(df_close_log)

# %% 
rolling_data(df_close_log)

# %% Biểu đồ phân rã chuỗi thời gian
decompose_data(df_close_log)

# %% Kiểm định tính dừng
print(adf_test(df_close_log))
print("------"*5 )
print(kpss_test(df_close_log))

# %% Kiểm định tự tương quan 
correlation_plot(df_close_log)

# %%
pacf(df_close_log)
acf(df_close_log)

# %% Chuyển đổi dữ liệu thành chuỗi dừng
diff = df_close_log.diff(1).dropna()

fig, ax = plt.subplots(2, sharex="all")
df_close_log.plot(ax=ax[0], title="Giá đóng cửa")
diff.plot(ax=ax[1], title="Sai phân bậc nhất")
plt.show() 

# %% Kiểm tra lại tính dừng của dữ liệu sau khi sai phân
print(adf_test(diff))
print("----------"*5)
print(kpss_test(diff))

pacf(diff)
acf(diff)

# %% 
window_size = 530  
test_size = 130
step_size = 500

# 
mse_list = []
rmse_list = []

plot_data_rolling_splits(df_close_log, window_size, test_size, step_size)

# %%
#%% 
for i in range(5):
    train_start = i * step_size
    train_end = train_start + window_size
    test_end = train_end + test_size

    if test_end > len(df_close_log):
        break

    train_data = df_close_log[train_start:train_end]
    test_data = df_close_log[train_end:test_end]

    print(f"Window {i + 1}:")
    print("TRAIN: ", train_data)
    print("TEST: ", test_data)

    # Huấn luyện mô hình AR
    model_ar = fit_ar_model(train_data, max_lag=5, criterion='aic')
    pred_ar = predict_ar_model(model_ar, train_data, test_data)
    evaluate_ar = evaluate_ar_model(test_data, pred_ar)
    plot_rolling_ar_pred(df_close_log, train_data, test_data, pred_ar, i)
    
    # Huấn luyện mô hình ARMA
    p = find_optimal_p(train_data, 5)
    q = find_optimal_q(train_data, 5)
    model_arma = fit_arma_model(train_data, p, q)
    fc_series_arma, lower_series_arma, upper_series_arma = predict_arma_model(model_arma, test_data)
    evaluate_arma = evaluate_arma_model(test_data, fc_series_arma)
    plot_rolling_arma_pred(df_close_log, train_data, test_data, fc_series_arma, lower_series_arma, upper_series_arma, i)

    # Huấn luyện mô hình ARIMA
    fitted_arima, order_arima = fit_arima_model(train_data)
    fc_series_arima, lower_series_arima, upper_series_arima = forecast_arima_model(fitted_arima, test_data)
    evaluate_arima = evaluate_arima_model(test_data, fc_series_arima)
    plot_rolling_arima_pred(df_close_log, train_data, test_data, fc_series_arima, lower_series_arima, upper_series_arima, i)
    
    # Huấn luyện mô hình SARIMA
    fitted_sarima = fit_sarima_model(train_data, seasonal_period=12)
    fc_series_sarima, lower_series_sarima, upper_series_sarima = predict_sarima_model(fitted_sarima, test_data)
    evaluate_sarima = evaluate_sarima_model(test_data, fc_series_sarima)
    plot_rolling_sarima_pred(df_close_log, train_data, test_data, fc_series_sarima, lower_series_sarima, upper_series_sarima, i)
    
    # Huấn luyện mô hình Holt-Winters
    fitted_hw = fit_hw_model(train_data, trend='add', seasonal=None, seasonal_periods=None)
    hw_forecast_series = predict_hw_model(fitted_hw, test_data)
    evaluate_hw = evaluate_hw_model(test_data, hw_forecast_series)
    plot_rolling_hw_pred(df_close_log, train_data, test_data, hw_forecast_series, i)
    
    # Tính toán các chỉ số đánh giá
    mse_ar = evaluate_ar['mse']
    rmse_ar = evaluate_ar['rmse']
    
    mse_arma = evaluate_arma['mse']
    rmse_arma = evaluate_arma['rmse']
    
    mse_arima = evaluate_arima['mse']
    rmse_arima = evaluate_arima['rmse']
    
    mse_sarima = evaluate_sarima['mse']
    rmse_sarima = evaluate_sarima['rmse']
    
    mse_hw = evaluate_hw['mse']
    rmse_hw = evaluate_hw['rmse']
    
    mse_list.append((mse_ar, mse_arma, mse_arima, mse_sarima, mse_hw))
    rmse_list.append((rmse_ar, rmse_arma, rmse_arima, rmse_sarima, rmse_hw))

    # Save model
    roll_save_model(model_ar, 'AR', i)
    roll_save_model(model_arma, 'ARMA', i)
    roll_save_model(fitted_arima, 'ARIMA', i)
    roll_save_model(fitted_sarima, 'SARIMA', i)
    roll_save_model(fitted_hw, 'Holt-Winters', i)

# %% 
print("\nEvaluation Metrics for Each Window in Rolling Window:")
for i, (mse, rmse) in enumerate(zip(mse_list, rmse_list)):
    print(f"Window {i+1}:")
    print(f"  AR MSE: {mse[0]}, RMSE: {rmse[0]}")
    print(f"  ARMA MSE: {mse[1]}, RMSE: {rmse[1]}")
    print(f"  ARIMA MSE: {mse[2]}, RMSE: {rmse[2]}")
    print(f"  SARIMA MSE: {mse[3]}, RMSE: {rmse[3]}")
    print(f"  Holt-Winters MSE: {mse[4]}, RMSE: {rmse[4]}")
    print("------"*10)

