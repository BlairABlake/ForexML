from backtesting import Backtest, Strategy
from datasets import ForexData
import torch
import numpy as np
from FNNwithNorm import FNNwithNorm

class SmaCross(Strategy):

    def init(self):
        self.close = self.data.Close
        self.i = 0
        self.model = FNNwithNorm(input_num=20).double()
        self.model.load_state_dict(torch.load("./model_states", map_location=torch.device("cpu")))

    def next(self):
        self.i += 1
        if(self.i < 20): return
        
        with torch.no_grad():
            data = self.close[self.i-20:self.i]
            prediction = self.model(torch.tensor(np.expand_dims(np.expand_dims(data, axis=0), axis=2))).numpy()[0]
        
        if not self.position:
            
            if prediction - self.close[self.i] >= 0.13: 
                self.buy()
                self.entry_i = self.i
            elif prediction - self.close[self.i] <= -0.13: 
                self.sell()
                self.entry_i = self.i
        
        else:
            if prediction - self.close[self.i] >= 0.13 and self.position.is_short and self.i - self.entry_i >= 10: 
                self.position.close()
            elif prediction - self.close[self.i] <= -0.13 and self.position.is_long and self.i - self.entry_i >= 10: 
                self.position.close()
        



dataset = ForexData("./data/USDJPY_H1.csv", header=0, normalize=False)
bt = Backtest(dataset.data.head(10000), SmaCross,
              cash=1000000, commission=.002, exclusive_orders=True)

output = bt.run()
print(output)
bt.plot()