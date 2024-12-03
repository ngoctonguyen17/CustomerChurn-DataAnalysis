# %% 
import pandas as pd
import numpy as np
import seaborn as sns
from function.TSA_Plot import *
from function.TSA_Fitmodel import *
from function.TSA_Evaluate import *
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

#%% Biểu đồ phân rã chuỗi thời gian
decompose_data(df_close_log)

#%% Kiểm định tính dừng
print(adf_test(df_close_log))
print("------"*5 )
print(kpss_test(df_close_log))

# %% Kiểm định tự tương quan 
correlation_plot(df_close_log)

#%%
pacf(df_close_log)
acf(df_close_log)

#%% Chuyển đổi dữ liệu thành chuỗi dừng
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

# %% Chia dữ liệu
train_data, test_data = df_close_log[:int(len(df_close_log)*0.8)], df_close_log[int(len(df_close_log)*0.8):]
plt.figure(figsize=(12, 6))
plt.plot(train_data, 'blue', label='Train data')
plt.plot(test_data, 'red', label='Test data')
plt.xlabel('Date')
plt.ylabel('Close prices')
plt.legend()
plt.show()

# %% AR Model
model_ar = fit_ar_model(train_data, max_lag=10, criterion='aic')
pred_ar = predict_ar_model(model_ar, train_data, test_data)
evaluate_ar = evaluate_ar_model(test_data, pred_ar)
plot_ar_pred(train_data, test_data, pred_ar)

# %% ARMA Model
p = find_optimal_p(train_data, 5)
q = find_optimal_q(train_data, 5)
model_arma = fit_arma_model(train_data, p, q)
fc_series_arma, lower_series_arma, upper_series_arma = predict_arma_model(model_arma, test_data)
evaluate_arma = evaluate_arma_model(test_data, fc_series_arma)
plot_arma_pred(train_data, test_data, fc_series_arma, lower_series_arma, upper_series_arma)

# %% ARIMA Model
fitted_arima, order_arima = fit_arima_model(train_data)
fc_series_arima, lower_series_arima, upper_series_arima = forecast_arima_model(fitted_arima, test_data)
evaluate_arima = evaluate_arima_model(test_data, fc_series_arima)
plot_arima_pred(train_data, test_data, fc_series_arima, lower_series_arima, upper_series_arima)

# %% SARIMA Model
fitted_sarima = fit_sarima_model(train_data, seasonal_period=12)
fc_series_sarima, lower_series_sarima, upper_series_sarima = predict_sarima_model(fitted_sarima, test_data)
evaluate_sarima = evaluate_sarima_model(test_data, fc_series_sarima)

plot_sarima_pred(train_data, test_data, fc_series_sarima, lower_series_sarima, upper_series_sarima)

# %% Holt-Winters Model
fitted_hw = fit_hw_model(train_data, trend='add', seasonal= None, seasonal_periods=None)
hw_forecast_series = predict_hw_model(fitted_hw, test_data)
evaluate_hw = evaluate_hw_model(test_data, hw_forecast_series)
plot_hw_pred(train_data, test_data, hw_forecast_series)

# %%
print("AR Model Evaluation:")
print(f"  - MSE:  {evaluate_ar['mse']:.4f}")
print(f"  - RMSE: {evaluate_ar['rmse']:.4f}")

print("ARMA Model Evaluation:")
print(f"  - MSE:  {evaluate_arma['mse']:.4f}")
print(f"  - RMSE: {evaluate_arma['rmse']:.4f}")

print("ARIMA Model Evaluation:")
print(f"  - MSE:  {evaluate_arima['mse']:.4f}")
print(f"  - RMSE: {evaluate_arima['rmse']:.4f}")

print("SARIMA Model Evaluation:")
print(f"  - MSE:  {evaluate_sarima['mse']:.4f}")
print(f"  - RMSE: {evaluate_sarima['rmse']:.4f}")

print("Holt-Winters Model Evaluation:")
print(f"  - MSE:  {evaluate_hw['mse']:.4f}")
print(f"  - RMSE: {evaluate_hw['rmse']:.4f}")
# %%
save_model(model_ar, 'AR')
save_model(model_arma, 'ARMA')
save_model(fitted_arima, 'ARIMA')
save_model(fitted_sarima, 'SARIMA')
save_model(fitted_hw, 'Holt-Winters')

