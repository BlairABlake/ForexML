from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG, EURUSD

import pandas as pd
import numpy as np

from datasets import ForexDataset

data = ForexDataset("./data/USDJPY_H1.csv", header=0)


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2) and np.random.rand() > 0.95:
            self.buy()
        elif crossover(self.ma2, self.ma1) and np.random.rand() > 0.95:
            self.sell()


bt = Backtest(data.data, SmaCross, commission=.002, cash=10000000,
              exclusive_orders=True)
stats = bt.run()
bt.plot()