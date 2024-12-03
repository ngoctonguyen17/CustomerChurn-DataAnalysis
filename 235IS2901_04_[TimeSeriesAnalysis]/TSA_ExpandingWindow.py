# %% 
import pandas as pd
import numpy as np
import seaborn as sns
from function.TSA_Fitmodel import *
from function.TSA_Evaluate import *
from function.TSA_Plot import *
from function.TSA_Function import *
from sklearn.model_selection import TimeSeriesSplit
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

# %% Kiểm định tính dừng
print(adf_test(df_close_log))
print("----"*10 )
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
def calculate_mean_metrics(mse_list, rmse_list):
    mean_mse = np.mean(mse_list, axis=0)
    mean_rmse = np.mean(rmse_list, axis=0)
    return mean_mse, mean_rmse

mean_mse_list = []
mean_rmse_list = []

for n_splits in range(2, 11):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    mse_list = []
    rmse_list = []

    for train_index, test_index in tscv.split(df_close_log):
        train_data, test_data = df_close_log.iloc[train_index], df_close_log.iloc[test_index]

        # Huấn luyện mô hình AR
        model_ar = fit_ar_model(train_data, max_lag=5, criterion='aic')
        pred_ar = predict_ar_model(model_ar, train_data, test_data)
        evaluate_ar = evaluate_ar_model(test_data, pred_ar)

        # Huấn luyện mô hình ARMA
        p = find_optimal_p(train_data, 5)
        q = find_optimal_q(train_data, 5)
        model_arma = fit_arma_model(train_data, p, q)
        fc_series_arma, lower_series_arma, upper_series_arma = predict_arma_model(model_arma, test_data)
        evaluate_arma = evaluate_arma_model(test_data, fc_series_arma)

        # Huấn luyện mô hình ARIMA
        fitted_arima, order_arima = fit_arima_model(train_data)
        fc_series_arima, lower_series_arima, upper_series_arima = forecast_arima_model(fitted_arima, test_data)
        evaluate_arima = evaluate_arima_model(test_data, fc_series_arima)

        # Huấn luyện mô hình SARIMA
        fitted_sarima = fit_sarima_model(train_data, seasonal_period=12)
        fc_series_sarima, lower_series_sarima, upper_series_sarima = predict_sarima_model(fitted_sarima, test_data)
        evaluate_sarima = evaluate_sarima_model(test_data, fc_series_sarima)

        # Huấn luyện mô hình Holt-Winters
        fitted_hw = fit_hw_model(train_data, trend='add', seasonal='add', seasonal_periods=12)
        hw_forecast_series = predict_hw_model(fitted_hw, test_data)
        evaluate_hw = evaluate_hw_model(test_data, hw_forecast_series)

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

    mean_mse, mean_rmse = calculate_mean_metrics(mse_list, rmse_list)
    mean_mse_list.append(mean_mse)
    mean_rmse_list.append(mean_rmse)

#%%
best_window_index = np.argmin([mean[2] for mean in mean_mse_list])  
best_n_splits = best_window_index + 2  

print(f"Best number of splits: {best_n_splits}")

# %% Chia dữ liệu 
tscv = TimeSeriesSplit(n_splits=best_n_splits)
mse_list = []
rmse_list = []
# %% 
for i, (train_index, test_index) in enumerate(tscv.split(df_close_log)):
    train_data, test_data = df_close_log.iloc[train_index], df_close_log.iloc[test_index]

    print(f"Fold {i + 1}:")
    print("TRAIN: ", train_data)
    print("TEST: ", test_data)

    # Huấn luyện mô hình AR
    model_ar = fit_ar_model(train_data, max_lag=5, criterion='aic')
    pred_ar = predict_ar_model(model_ar, train_data, test_data)
    evaluate_ar = evaluate_ar_model(test_data, pred_ar)
    plot_expand_ar_pred(df_close_log, train_data, test_data, pred_ar, i)


    # Huấn luyện mô hình ARMA
    p = find_optimal_p(train_data, 5)
    q = find_optimal_q(train_data, 5)
    model_arma = fit_arma_model(train_data, p, q)
    fc_series_arma, lower_series_arma, upper_series_arma = predict_arma_model(model_arma, test_data)
    evaluate_arma = evaluate_arma_model(test_data, fc_series_arma)
    plot_expand_arma_pred(df_close_log, train_data, test_data, fc_series_arma, lower_series_arma, upper_series_arma, i)
    
    # Huấn luyện mô hình ARIMA
    fitted_arima, order_arima = fit_arima_model(train_data)
    fc_series_arima, lower_series_arima, upper_series_arima = forecast_arima_model(fitted_arima, test_data)
    evaluate_arima = evaluate_arima_model(test_data, fc_series_arima)
    plot_expand_arima_pred(df_close_log, train_data, test_data, fc_series_arima, lower_series_arima, upper_series_arima, i)
    
    # Huấn luyện mô hình SARIMA
    fitted_sarima = fit_sarima_model(train_data, seasonal_period=12)
    fc_series_sarima, lower_series_sarima, upper_series_sarima = predict_sarima_model(fitted_sarima, test_data)
    evaluate_sarima = evaluate_sarima_model(test_data, fc_series_sarima)
    plot_expand_sarima_pred(df_close_log, train_data, test_data, fc_series_sarima, lower_series_sarima, upper_series_sarima, i)
    
    # Huấn luyện mô hình Holt-Winters
    fitted_hw = fit_hw_model(train_data, trend='add', seasonal=None, seasonal_periods=None)
    hw_forecast_series = predict_hw_model(fitted_hw, test_data)
    evaluate_hw = evaluate_hw_model(test_data, hw_forecast_series)
    plot_expand_hw_pred(df_close_log, train_data, test_data, hw_forecast_series, i)
    
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
    
    # Lưu trữ các chỉ số đánh giá
    mse_list.append((mse_ar, mse_arma, mse_arima, mse_sarima, mse_hw))
    rmse_list.append((rmse_ar, rmse_arma, rmse_arima, rmse_sarima, rmse_hw))
    
    # Lưu các mô hình
    ex_save_model(model_ar, 'AR', i)
    ex_save_model(model_arma, 'ARMA', i)
    ex_save_model(fitted_arima, 'ARIMA', i)
    ex_save_model(fitted_sarima, 'SARIMA', i)
    ex_save_model(fitted_hw, 'Holt-Winters', i)

# %% 
print("\nEvaluation Metrics for Each Window:")
for i, (mse, rmse) in enumerate(zip(mse_list, rmse_list)):
    print(f"Window {i+1}:")
    print(f"  AR MSE: {mse[0]}, RMSE: {rmse[0]}")
    print(f"  ARMA MSE: {mse[1]}, RMSE: {rmse[1]}")
    print(f"  ARIMA MSE: {mse[2]}, RMSE: {rmse[2]}")
    print(f"  SARIMA MSE: {mse[3]}, RMSE: {rmse[3]}")
    print(f"  Holt-Winters MSE: {mse[4]}, RMSE: {rmse[4]}")


