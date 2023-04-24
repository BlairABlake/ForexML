import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

class ForexData(Dataset):
    order_mapping = {
        "ohlc": ["Open", "High", "Low", "Close"],
        "o": ["Open"],
        "h": ["High"],
        "l": ["Low"],
        "c": ["Close"]
    }
    def __init__(self, data_file, header=-1, sep="\t", data_order="ohlc", normalize=True):
        if header != -1:
            self._data =  pd.read_csv(data_file, header=header, sep=sep)
        else:
            self._data = pd.read_csv(data_file, names=["date", "Open", "High", "Low", "Close"], sep=sep)

        if normalize:
            self.scaler = StandardScaler().fit(self._data[self.order_mapping["ohlc"]])
            self._data[self.order_mapping["ohlc"]] = self.scaler.transform(self._data[self.order_mapping["ohlc"]])

        self.data_order = data_order

        self.data = self._data[self.order_mapping[self.data_order]]
    
    def __len__(self):
        return self.data.shape[0] - 1

    def __getitem__(self, index):
        return self.data.iloc[index].to_numpy(), self.data.iloc[index + 1].to_numpy()
    
class ForexDataLinregressMixin(ForexData):
    def add_lineregress(self, window=10):
        self.window = window

        self.data["Slope"] = self.data["Close"].rolling(window).apply(lambda x: linregress(self._linregress_data(x)).slope)
        self.data = self.data.dropna().reset_index(drop=True)
    
    def _linregress_data(self, data):
        return list(zip(range(len(data)), data))

class ForexDataWithInterval(ForexData):
    def __init__(self, data_file, header=-1, sep="\t", data_order="ohlc", normalize=True, input_duration=10, output_duration=10, interval=0):
        super().__init__(data_file, header, sep, data_order, normalize=normalize)
        self.data_order = data_order
        self.input_duation = input_duration
        self.output_duration = output_duration
        self.interval = interval
    
    def __len__(self):
        return self.data.shape[0] - (self.input_duation+self.interval+self.output_duration)

    def __getitem__(self, index):
        return \
            self.data.iloc[index:index+self.input_duation].to_numpy(), \
            self.data.iloc[index+self.input_duation+self.interval:index+self.input_duation+self.interval+self.output_duration].to_numpy()
        
class ForexDataLinregresswithInterval(ForexDataWithInterval, ForexDataLinregressMixin):

    def __init__(self, data_file, header=-1, sep="\t", data_order="ohlc", input_duration=10, output_duration=10, interval=0):
        super().__init__(data_file, header, sep, data_order, input_duration, output_duration, interval)

        self.add_lineregress(window=input_duration)
    
    def __getitem__(self, index):
        x,y =  super().__getitem__(index)
        return x, y[0][0]
    
class ForexPricePredictionDataset(ForexDataWithInterval):
    def __getitem__(self, index):
        x, t = super().__getitem__(index)
        return \
            x[:, 0], \
            t[:, 0]
        
    