# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 读取数据
try:
    df1 = pd.read_excel('data.xlsx', sheet_name=0)  # 0对应第一支股票
    df1 = df1.iloc[3600:-10, 1:]  # 选取从第3600行开始的数据，大概是2006年一月
except ImportError:
    print("请先安装 openpyxl: pip install openpyxl")
    
# 数据归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
df0 = min_max_scaler.fit_transform(df1)
df = pd.DataFrame(df0, columns=df1.columns)
X = df.iloc[:, :-1]
y = df['target']
print("X shape:", X.shape)
print("y shape:", y.shape)

# 构造训练集测试集
y = pd.DataFrame(y.values, columns=['goal'])
x = X
cut = 10  # 取最后10天为测试集
X_train, X_test = x.iloc[:-cut], x.iloc[-cut:]
y_train, y_test = y.iloc[:-cut], y.iloc[-cut:]
X_train, X_test = X_train.values, X_test.values
y_train, y_test = y_train.values, y_test.values

# 建立BP模型并训练
model = Sequential([
    Dense(16, activation='relu', kernel_initializer='uniform', input_dim=10),
    Dense(4, activation='sigmoid', kernel_initializer='uniform'),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=256)

# 输出模型结构
model.summary()

# 训练集预测
y_train_predict = model.predict(X_train)
y_train_predict = y_train_predict[:, 0]

# 绘制训练集结果
plt.figure(figsize=(12, 6))
draw = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_train_predict)], axis=1)
draw.iloc[100:400, 0].plot()
draw.iloc[100:400, 1].plot()
plt.legend(('real', 'predict'), fontsize='15')
plt.title("Train Data", fontsize='30')
plt.show()

# 测试集预测
y_test_predict = model.predict(X_test)
y_test_predict = y_test_predict[:, 0]

# 绘制测试集结果
plt.figure(figsize=(12, 6))
draw = pd.concat([pd.DataFrame(y_test), pd.DataFrame(y_test_predict)], axis=1)
draw.iloc[:, 0].plot()
draw.iloc[:, 1].plot()
plt.legend(('real', 'predict'), loc='upper right', fontsize='15')
plt.title("Test Data", fontsize='30')
plt.show()

# 计算评估指标
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

print('训练集上的 MAE/MSE/MAPE:')
print(f"MAE: {mean_absolute_error(y_train_predict, y_train):.4f}")
print(f"MSE: {mean_squared_error(y_train_predict, y_train):.4f}")
print(f"MAPE: {mape(y_train_predict, y_train):.4f}%")

print('\n测试集上的 MAE/MSE/MAPE:')
print(f"MAE: {mean_absolute_error(y_test_predict, y_test):.4f}")
print(f"MSE: {mean_squared_error(y_test_predict, y_test):.4f}")
print(f"MAPE: {mape(y_test_predict, y_test):.4f}%")

# 计算预测涨跌准确率
y_var_test = y_test[1:] - y_test[:-1]
y_var_predict = y_test_predict[1:] - y_test_predict[:-1]
accuracy = np.mean(np.sign(y_var_test) == np.sign(y_var_predict))
print(f'\n预测涨跌正确率: {accuracy:.2%}')