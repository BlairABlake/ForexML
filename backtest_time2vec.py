from backtesting import Backtest, Strategy
from datasets import ForexData
import torch
import numpy as np
import pandas as pd
from FNNwithNorm import FNNwithNorm
import models.time2vec as time2vec

class CustomStrategy(Strategy):

    def init(self):
        self.i = 0
        self.data = self._data.df

    def next(self):
        _, _, _, _, up, down, staionary = self.data.iloc[self.i]

        if not self.position:
            if up >= 0.7:
                self.buy()
            elif down >= 0.7:
                self.down()
        else:
            if self.position.is_long and down >= 0.3:
                self.position.close()
            if self.position.is_short and up >= 0.3:
                self.position.close()

        self.i += 1
        



data = pd.read_table("./data/AUDJPY_H1.csv", header=0)
data["Time"] = pd.to_datetime(data["Time"])
data = data.set_index("Time")
bt = Backtest(data.head(1000), CustomStrategy,
              cash=1000000, commission=.002, exclusive_orders=True)

output = bt.run()
print(output)
bt.plot()