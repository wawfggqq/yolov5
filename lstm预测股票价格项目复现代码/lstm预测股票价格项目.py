import pandas_datareader.data as web
import datetime
start= datetime.datetime(2000,1,1)
end=datetime.datetime(2021,9,1)
df = web.DataReader('GOOGL', 'stooq', start, end)
pre_days = 10
def Stock_Price_LSTM_Data_Precessing(df,mem_his_days,predays):
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    df['label'] = df['Close'].shift(-pre_days)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    sca_X = scaler.fit_transform(df.iloc[:, :-1])
    mem_his_days = 5

    from collections import  deque
    deq = deque(maxlen=mem_his_days)

    X = []
    for i in sca_X:
        deq.append(list(i))
        if len(deq) == mem_his_days:
            X.append(list(deq))

    X_lately = X[-pre_days:]
    X = X[:-pre_days]

    y = df['label'].values[mem_his_days-1:-pre_days]

    import numpy as np
    X=np.array(X)
    y=np.array(y)

    return X,y,X_lately

X,y,X_lately = Stock_Price_LSTM_Data_Precessing(df,mem_his_days=5,predays=10)
print(len(X))
print(len(y))
print(len(X_lately))

#开始在建模中使用、构建这个模型
mem_days=[5]
lstm_layers=[1]
dense_layers = [1]
units = [32]

from tensorflow.keras.callbacks import ModelCheckpoint
for the_mem_days in mem_days:
    for the_lstm_layers in lstm_layers:
        for the_dense_layers in dense_layers:
            for the_units_layers in units:
                filepath = './model/{epoch:02d}-{val_mape:.2f}' + f'men_{the_lstm_layers}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_units_{the_units_layers}.keras'
                checkpoint = ModelCheckpoint(
                    filepath=filepath,
                    save_weights_only=False,
                    monitor='val_mape',
                    save_best_only=True,
                    mode='min')
                X,y,X_lately = Stock_Price_LSTM_Data_Precessing(df,mem_his_days=the_mem_days,predays=10)
                from sklearn.model_selection import train_test_split
                X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False, test_size=0.1)


                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, Dropout, LSTM
                model = Sequential()
                model.add(LSTM(10,input_shape=X.shape[1:],activation='relu',return_sequences=True))
                model.add(Dropout(0.1))

                for i in range(the_lstm_layers):
                    model.add(LSTM(the_units_layers,activation='relu',return_sequences=True))
                    model.add(Dropout(0.1))

                model.add(LSTM(the_units_layers,activation='relu'))
                model.add(Dropout(0.1))

                for i in range(the_dense_layers):
                    model.add(Dense(the_units_layers,activation='relu'))
                    model.add(Dropout(0.1))

                model.add(Dense(1))
                model.compile(optimizer='adam',
                              loss='mse',
                              metrics=['mape'])
                model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),callbacks=[checkpoint])

from tensorflow.keras.models import load_model
best_model = load_model('./model/54-5.14men_1_lstm_1_dense_1_units_32.keras')
pre = best_model.predict(X_test)
best_model.summary()

best_model.evaluate(X_test,y_test)
model.evaluate(X_test,y_test)


import matplotlib.pyplot as plt
df_time=df.index[-len(y_test):]

plt.plot(df_time,y_test,color='red',label='price')
plt.plot(df_time,pre,color='green',label='predict')
plt.show()

print(len(y_test))