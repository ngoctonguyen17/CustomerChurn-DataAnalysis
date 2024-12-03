#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

#%%
df = pd.read_csv('data/Restaurant_revenue (1).csv')

df.info()
df.head()

#%%
df.isnull().sum()
#%%
df.duplicated().sum()
#%%
print(df)

#%%
df.describe()
#%% Ngưỡng trên của tập dữ liệu
max_thresold = df["Monthly_Revenue"].quantile(0.99)
max_thresold

#%% Ngưỡng dưới của tập dữ liệu
min_thresold = df["Monthly_Revenue"].quantile(0.01)
min_thresold
#%%
df[df['Monthly_Revenue']>max_thresold]
#%%
df[df['Monthly_Revenue']<min_thresold]
#%%Loại bỏ giá trị ngoại lai
df_cleaned = df[(df["Monthly_Revenue"] >= min_thresold) & (df["Monthly_Revenue"] <= max_thresold)]
df_cleaned.head()  
print(f"Số lượng quan sát trong df: {len(df)}")
print(f"Số lượng quan sát trong df_cleaned: {len(df_cleaned)}")
#%%
mapping = {'Japanese': 0, 'American': 1, 'Mexican': 2, 'Italian':3}
df_cleaned['Cuisine_Type'] = df['Cuisine_Type'].map(mapping)
df_cleaned.head()

# %%
sns.pairplot(df_cleaned)
plt.show()
# %%Biểu đồ heatmap 
#using seaborn
import seaborn as sns
import matplotlib.pyplot as plt


corr = df_cleaned.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True)
plt.show()

# %%
def plot_boxplot(df_cleaned, feature_name):
    # Boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(df_cleaned[feature_name])
    plt.xlabel(feature_name)
    plt.title(f"Boxplot of {feature_name}")
    plt.show()


def plot_scatter(df_cleaned, feature_name):
    # Scatter plot
    plt.scatter(df_cleaned[feature_name], df_cleaned["Monthly_Revenue"])
    plt.xlabel(feature_name)
    plt.ylabel("Monthly_Revenue")
    plt.title(f"Scatter plot of {feature_name} vs Monthly_Revenue")
    plt.show()


# %% - biến Number_of_Customers
plot_boxplot(df_cleaned, "Number_of_Customers")
plot_scatter(df, "Number_of_Customers")

# %% biến Menu_Price
plot_boxplot(df_cleaned, "Menu_Price")
plot_scatter(df_cleaned, "Menu_Price")

# %% - biến Marketing
plot_boxplot(df_cleaned, "Marketing_Spend")
plot_scatter(df_cleaned, "Marketing_Spend")

# %% - biểu đồ boxplot cho biến revenue
plot_boxplot(df_cleaned, "Monthly_Revenue")

#%%
X = df_cleaned[["Number_of_Customers", "Menu_Price", "Marketing_Spend"]]
y = df_cleaned[["Monthly_Revenue"]]


# %%

def cal_vif(df, dependent_var, excluded_vars=[]):
    X = df.drop(columns=[dependent_var] + excluded_vars)
    X['Intercept'] = 1
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

X['Intercept'] = 1
# Tính toán VIF cho từng biến
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
#%%
#XÂY DỰNG MÔ HÌNH HỒI QUY ĐƠN BIẾN

#%% Chọn biến độc lập và biến phụ thuộc
X = df_cleaned[["Number_of_Customers"]]
y = df_cleaned[["Monthly_Revenue"]]

# %% - Model
LG = LinearRegression()


# %% - Sử dụng k-fold cross-validation với k=5
def kfold_cross_validation(LG, X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2_scores = []
    mse_scores = []
    mae_scores = []
    rmse_scores = []
    fold_count = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        LG.fit(X_train, y_train)
        y_pred = LG.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print(f"Fold {fold_count}:")
        print("R-squared (R^2):", r2)
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print()

        fold_count += 1

    print("Kết quả kiểm định K-fold bằng Python")
    print("Trung bình R-squared (R^2) cho cả 5 fold:", np.mean(r2_scores))
    print("Trung bình MAE cho cả 5 fold:", np.mean(mae_scores))
    print("Trung bình MSE cho cả 5 fold:", np.mean(mse_scores))
    print("Trung bình RMSE cho cả 5 fold:", np.mean(rmse_scores))

kfold_cross_validation(LG, X, y)

# %%
# Thêm intercept vào dữ liệu X
X = sm.add_constant(X)
# Xây dựng mô hình hồi quy tuyến tính bằng phương pháp OLS
model = sm.OLS(y, X).fit()
# In ra bảng OLS
print(model.summary())




#%% Chia tập 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2)

# %%
# Thêm intercept vào dữ liệu X
X_train = sm.add_constant(X_train)
# Xây dựng mô hình hồi quy tuyến tính bằng phương pháp OLS
model = sm.OLS(y_train, X_train).fit()
# In ra bảng OLS
print(model.summary())


#%%
# Dự đoán giá trị revenue từ mô hình
X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("R-squared (R^2):", r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)


#%%
# Vẽ biểu đồ so sánh giữa actual revenue và predicted revenue
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')  # Đường thẳng y=x
plt.title('Actual revenue vs. Predicted revenue')
plt.xlabel('Actual revenue')
plt.ylabel('Predicted revenue')
plt.grid(True)
plt.show()
# %% -  Save Model
import pickle


# Hàm để lưu model dùng pickle
def save_model_with_pickle(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# %%
# Lưu model đơn biến
filename_single = 'linear_single_variable_model.pkl'
save_model_with_pickle(model, filename_single)
print(f"Model đơn biến đã được lưu trong file: {filename_single}")


# %%
#XÂY DỰNG MÔ HÌNH HỒI QUY ĐA BIẾN

#%% Chọn biến độc lập và biến phụ thuộc
X = df_cleaned[["Number_of_Customers", "Menu_Price", "Marketing_Spend"]]
y = df_cleaned[["Monthly_Revenue"]]

# %% - Model
LG = LinearRegression()


# %% - Sử dụng k-fold cross-validation với k=5
def kfold_cross_validation(LG, X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2_scores = []
    mse_scores = []
    mae_scores = []
    rmse_scores = []
    fold_count = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        LG.fit(X_train, y_train)
        y_pred = LG.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print(f"Fold {fold_count}:")
        print("R-squared (R^2):", r2)
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print()

        fold_count += 1

    print("Kết quả kiểm định K-fold bằng Python")
    print("Trung bình R-squared (R^2) cho cả 5 fold:", np.mean(r2_scores))
    print("Trung bình MAE cho cả 5 fold:", np.mean(mae_scores))
    print("Trung bình MSE cho cả 5 fold:", np.mean(mse_scores))
    print("Trung bình RMSE cho cả 5 fold:", np.mean(rmse_scores))


# %%
kfold_cross_validation(LG, X, y)

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2)

# %%
# Thêm intercept vào dữ liệu X
X_train = sm.add_constant(X_train)
# Xây dựng mô hình hồi quy tuyến tính bằng phương pháp OLS
model = sm.OLS(y_train, X_train).fit()
# In ra bảng OLS
print(model.summary())

# %%
# Dự đoán giá trị revenue từ mô hình
#%%
# Dự đoán giá trị revenue từ mô hình
X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("R-squared (R^2):", r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
#%%
# Vẽ biểu đồ so sánh giữa actual revenue và predicted revenue
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')  # Đường thẳng y=x
plt.title('Actual revenue vs. Predicted revenue')
plt.xlabel('Actual revenue')
plt.ylabel('Predicted revenue')
plt.grid(True)
plt.show()



# %%
# Lưu model đabiến
filename_single = 'linear_multiple_variable_model.pkl'
save_model_with_pickle(model, filename_single)
print(f"Model đa biến đã được lưu trong file: {filename_single}")

# %%
