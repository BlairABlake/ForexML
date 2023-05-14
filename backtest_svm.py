from backtesting import Backtest, Strategy
from datasets import ForexData
import torch
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from tqdm import tqdm

class CustomStrategy(Strategy):
    mapping = ["down", "stationary", "up"]
    def init(self):
        self.i = 0
        self.m_data = self._data.df
        file = np.load("time2vec_udsjpy.npz")
        self.files, self.vectors = file["arr_0"], file["arr_1"]
        self.files = np.array([datetime.strptime(f, "%Y-%m-%d %H_%M_%S") for f in self.files])
        with open("svm.pkl", mode="rb") as f:
            self.model = pickle.load(f)
        self.pbar = tqdm(total=len(self._data))

    def next(self):
        i = np.where(self.m_data.iloc[self.i].name == self.files)[0]
        if i.size == 0: return
        else: i = i[0]
        prediction = self.mapping[self.model.predict([self.vectors[i]])[0]]

        if prediction == "staioanry": return

        if prediction == "up" and not self.position.is_long:
            self.buy()
        elif prediction == "down" and not self.position.is_short:
            self.sell()

        self.pbar.update(1)
        self.i += 1
        



dataset = ForexData("data/USDJPY_H1.csv", normalize=False, data_order="tohlc", header=0)
data = dataset.data
bt = Backtest(data.head(1000), CustomStrategy,
              cash=1000000, commission=.002, exclusive_orders=True)

output = bt.run()
print(output)
bt.plot()