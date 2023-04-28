import yfinance as yf
import torch
import numpy as np
from FNNwithNorm import FNNwithNorm
import matplotlib.pyplot as plt

model = FNNwithNorm(input_num=20).double()
model.load_state_dict(torch.load("./model_states", map_location=torch.device("cpu")))
ticker = yf.Ticker("USDJPY=X")
data = ticker.history(interval="1h", start="2023-04-20", end="2023-04-25")
data_original = data["Close"]
data = data["Close"].to_numpy()
with torch.no_grad():
    predictions = data_original.rolling(window=20).apply(lambda s: model(torch.tensor(np.expand_dims(np.expand_dims(s.to_numpy(), axis=0), axis=2))).numpy()[0]).to_numpy()



plt.plot(data)
plt.plot(predictions)
plt.show()