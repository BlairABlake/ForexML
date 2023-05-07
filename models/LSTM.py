import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D
from keras.callbacks import EarlyStopping

import os

_ohlc = set(["Open", "High", "Low", "Close"])

def check_if_fx_data(data):
    if _ohlc.issubset(set(data.columns)):
        return True
    return False

def load_fx_data(path, table=False):
    if table:
        data = pd.read_table(path)
    else:
        data = pd.read_csv(path)
    if not check_if_fx_data(data): raise Exception("Given data is not valid data type")
    data['Open'] = data['Open'].astype(float)
    data['Close'] = data['Close'].astype(float)
    data['High'] = data['High'].astype(float)
    data['Low'] = data['Low'].astype(float)

    return data

def add_features(data):
    if not check_if_fx_data(data): raise Exception("Given data is not valid data type")
    data['SA_OC'] = data['Close'] - data['Open'] 
    data['MA_40'] = data['Close'].rolling(window=40).mean() 
    data['High_MA_Deviation'] = ((data['High'] - data['MA_40']) / data['High']).rolling(window=9).sum().round(4)
    data['Low_MA_Deviation'] = ((data['Low'] - data['MA_40']) /  data['High']).rolling(window=9).sum().round(4)

    return data

def preprocessing(data):
    if not check_if_fx_data(data): raise Exception("Given data is not valid data type")
    data = data.dropna()
    numeric_data = data[['Close', 'SA_OC', 'High_MA_Deviation', 'Low_MA_Deviation']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    numeric_data = scaler.fit_transform(numeric_data)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(data["Close"].values.reshape(-1, 1))

    return close_scaler, numeric_data

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i])
        Y.append(dataset[i+time_step, 1])
    return np.array(X), np.array(Y)

def get_fxlstm(feature_num, output_num):
    return Sequential([
        Dense(1024, activation="relu", input_shape=(feature_num,)),
        Dense(1024, activation="relu"),
        Dense(output_num)
    ])
    
def train_fxlstm(X, Y, batch_size, epochs):
    model = get_fxlstm(X.shape[1], TIME_STEPS)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.fit(X, Y, batch_size=batch_size, epochs=epochs)
    return model

def predict(model, X, scaler):
    predictions = model.predict(X)
    return scaler.inverse_transform(predictions.reshape(-1, 1))

def multi_step_predict(model, input_data, n_steps):
    predictions = []
    current_input = input_data.copy()

    for _ in range(n_steps):
        prediction = model.predict(current_input)
        predictions.append(prediction[0][0])

        current_input = np.roll(current_input, -1)
        current_input[-1] = prediction

    return np.array(predictions)

def predict_from_raw_data(model, raw_data, time_steps=100, num=100, date_col="Time"):
    data = add_features(raw_data)
    date = data[date_col]
    scaler, data = preprocessing(data)
    X, _ = create_dataset(data, time_step=time_steps)
    predictions = multi_step_predict(model, np.expand_dims(X[2], axis=0), num)
    predictions = scaler.inverse_transform(predictions)

    return pd.DataFrame({"Time": date[-num:], "predictions": predictions.flatten()})



TIME_STEPS = 100
BATCH_SIZE = 64
EPOCHS = 10

data = load_fx_data("./data/USDJPY_H1.csv", table=True)
data = add_features(data)
scaler, data = preprocessing(data)
feature_num = data.shape[1]
X, Y = create_dataset(data, TIME_STEPS)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
callbacks_list = []
callbacks_list.append(early_stopping)

if os.path.exists("./fxlstm.h5"):
    model = get_fxlstm(feature_num, TIME_STEPS)
    model.load_weights("./fxlstm.h5")
else:
    model = train_fxlstm(X_train, Y_train, BATCH_SIZE, EPOCHS)

predictions = predict(model, X_test, scaler)
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

# plt.plot(Y_test, label='Actual')
# plt.plot(predictions, label='Predicted')
# plt.xlabel('t')
# plt.ylabel('Exchange Rate')
# plt.legend()
# plt.show()

model.save_weights("fxlstm.h5")

new_data = load_fx_data("JPY_H1_2301-2303.csv")
new_predictions = predict_from_raw_data(model, new_data, TIME_STEPS)

new_data = new_data.set_index("Time")
new_predictions = new_predictions.set_index("Time")
result = pd.merge(new_data, new_predictions, how='inner', left_index=True, right_index=True)[["Close", "predictions"]]

plt.plot(result["Close"], label="Actual")
plt.plot(result["predictions"], label="predicted")
plt.show()