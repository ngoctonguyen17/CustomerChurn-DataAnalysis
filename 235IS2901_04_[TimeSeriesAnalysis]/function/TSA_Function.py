#%%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import pickle
import os
#%%
def adf_test(data):
    indices = ["ADF: Test statistic", "p value", "# of Lags", "# of Observations"]
    test = adfuller(data, autolag="AIC")
    results = pd.Series(test[0:4], index=indices)
    for key, value in test[4].items():
        results[f"Critical Value ({key})"] = value

    if results[1] <= 0.05:
        print("Reject the null hypothesis (H0), \nthe data is stationary")
    else:
        print("Fail to reject the null hypothesis (H0), \nthe data is non-stationary")

    return results

def kpss_test(data):
    indices = ["KPSS: Test Statistic", "p value", "# of Lags"]
    test = kpss(data)
    results = pd.Series(test[0:3], index=indices)
    for key, value in test[3].items():
        results[f"Critical Value ({key})"] = value
    if results[1] <= 0.05:
        print("Reject the null hypothesis (H0), \nthe data is non-stationary")
    else:
        print("Fail to reject the null hypothesis (H0), \nthe data is stationary")
         
    return results

def rolling_data(data):
    plt.figure(figsize=(12, 6))
    rolmean = data.rolling(12).mean()
    rolstd = data.rolling(12).std()
    plt.plot(data, color="blue", label="Original")
    plt.plot(rolmean, color="red", label="Rolling Mean")
    plt.plot(rolstd, color="black", label="Rolling Std")
    plt.title("Rolling Mean and Standard Deviation")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def decompose_data(data):
    plt.figure(figsize=(12, 8))
    decompose_result = seasonal_decompose(data, model='multiplicative', period=30)
    decompose_result.plot()
    plt.suptitle("Seasonal Decomposition")
    plt.show()

def configuration():
    plt.rcParams['figure.figsize'] = (10,8)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 16

def find_optimal_p(data, max_lag):
    results = []
    for lag in range(1, max_lag + 1):
        model = AutoReg(data, lags=lag)
        model_fitted = model.fit()
        results.append((lag, model_fitted.aic))
    
    best_p = min(results, key=lambda x: x[1])[0]
    print("Optimal p value:", best_p)
    return best_p

def find_optimal_q(data, max_q):
    results = []
    for q in range(1, max_q + 1):
        model = ARIMA(data, order=(0, 0, q))
        model_fitted = model.fit()
        results.append((q, model_fitted.aic))
    
    best_q = min(results, key=lambda x: x[1])[0]
    print("Optimal q value:", best_q)
    return best_q

def save_model(model, model_name):
    folder_path = f'models/{model_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f'{model_name}-SimpleSplit.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    
def ex_save_model(model, model_name, fold):
    folder_path = f'models/{model_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f'{model_name}{fold + 1}-Expanding.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def roll_save_model(model, model_name, fold):
    folder_path = f'models/{model_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f'{model_name}{fold + 1}-Rolling.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)