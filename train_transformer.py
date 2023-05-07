import lightning.pytorch as pl
from torch.utils.data import DataLoader
from models.transformer import Transformer
from datasets import ForexPricePredictionDataset
import yfinance as yf

input_num = 20
output_num = 1
batch_size = 128
model = Transformer(input_num,output_num, 4).double()
dataset = ForexPricePredictionDataset("./data/USDJPY_H1.csv", header=0, data_order="ohlc", input_duration=input_num, output_duration=output_num)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
trainer = pl.Trainer(max_epochs=10, enable_progress_bar=True, accelerator="cpu", default_root_dir="./checkpoints")
trainer.fit(model, train_dataloaders=dataloader)