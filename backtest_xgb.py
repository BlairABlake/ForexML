from backtesting import Backtest, Strategy
from datasets import ForexData
import torch
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from tqdm import tqdm
import xgboost as xgb

class CustomStrategy(Strategy):
    mapping = ["down", "stationary", "up"]
    def init(self):
        self.i = 0
        self.m_data = self._data.df
        file = np.load("time2vec_udsjpy.npz")
        self.files, self.vectors = file["arr_0"], file["arr_1"]
        self.files = np.array([datetime.strptime(f, "%Y-%m-%d %H_%M_%S") for f in self.files])
        self.model = xgb.XGBClassifier()
        self.model.load_model("xgb.model")
        self.pbar = tqdm(total=len(self._data))

    def next(self):
        self.pbar.update(1)
        self.i += 1
        i = np.where(self.m_data.iloc[self.i].name == self.files)[0]
        if i.size == 0: return
        else: i = i[0]
        prediction = self.mapping[self.model.predict([self.vectors[i]])[0]]

        if not self.position:
            if prediction=="up":
                self.buy(size=1000)
            elif prediction=="down":
                self.sell(size=1000)
        else:
            if self.position.is_long and prediction=="down":
                self.position.close()
            if self.position.is_short and prediction=="up":
                self.position.close()

dataset = ForexData("data/USDJPY_H1.csv", normalize=False, data_order="tohlc", header=0)
data = dataset.data
bt = Backtest(data.head(1000), CustomStrategy,
              cash=1000000, commission=.00002, exclusive_orders=True, trade_on_close=True)

output = bt.run()
output.to_csv("xgb_backtest_result.csv")
output["_trades"].to_csv("xgb_backtest_trades.csv")
bt.plot()
