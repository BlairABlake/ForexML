from backtesting import Backtest, Strategy
from datasets import ForexData
import torch
import numpy as np
import pandas as pd
from FNNwithNorm import FNNwithNorm
import time2vec

class CustomStrategy(Strategy):

    def init(self):
        self.i = 0
        self.model = FNNwithNorm(input_num=20).double()

    def next(self):
        self.i += 1
        if(self.i < 30): return
        
        with torch.no_grad():
            data = self.data.df.iloc[self.i-30:self.i]
            prediction = time2vec.predict(data, checkpoint_path="epoch=36-step=2590.ckpt")
            label = time2vec.pred2label(prediction)
            
        
        if not self.position:
            
            if label == "up": 
                self.buy()
                self.entry_i = self.i
            elif label == "down": 
                self.sell()
                self.entry_i = self.i
        
        else:
            if label == "up" and self.position.is_short and self.i - self.entry_i >= 10: 
                self.position.close()
            elif label =="down" and self.position.is_long and self.i - self.entry_i >= 10: 
                self.position.close()
        



dataset = ForexData("./data/USDJPY_H1.csv", header=0, normalize=False, data_order="tohlc")
data = dataset.data
data["Time"] = pd.to_datetime(data["Time"])
data = data.set_index("Time")
bt = Backtest(data.head(1000), CustomStrategy,
              cash=1000000, commission=.002, exclusive_orders=True)

output = bt.run()
print(output)
bt.plot()