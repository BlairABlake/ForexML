import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ForexData(Dataset):
    order_mapping = {
        "ohlc": ["Open", "High", "Low", "Close"],
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close"
    }
    def __init__(self, data_file, header=-1, sep="\t", data_order="ohlc", input_duration=10, output_duration=10, interval=0):
        if header != -1:
            self.data = pd.read_csv(data_file, header=header, sep=sep)
        else:
            self.data = pd.read_csv(data_file, names=["date", "Open", "High", "Low", "Close"], sep=sep)

        self.data_order = data_order
        self.input_duation = input_duration
        self.output_duration = output_duration
        self.interval = interval
    
    def __len__(self):
        return self.data.shape[0] - (self.input_duation+self.interval+self.output_duration)

    def __getitem__(self, index):
        return (
            self.data.iloc[index:index+self.input_duation][self.order_mapping[self.data_order]].to_numpy(), 
            self.data.iloc[index+self.input_duation+self.interval:index+self.input_duation+self.interval+self.output_duration][self.order_mapping[self.data_order]].to_numpy()
        )
    
class ForexPricePredictionDataset(ForexData, Dataset): pass