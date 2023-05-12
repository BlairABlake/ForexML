import lightning.pytorch as pl
from torch.utils.data import DataLoader
from models.NormNet import NormNet
from datasets import ForexPricePredictionDataset

input_num = 60
output_num = 10
batch_size = 256
model = NormNet(input_num,output_num).double()
dataset = ForexPricePredictionDataset("./data/USDJPY_H1.csv", header=0, data_order="c", input_duration=input_num, output_duration=output_num, normalize=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
trainer = pl.Trainer(max_epochs=10, enable_progress_bar=True, accelerator="cpu", default_root_dir="./checkpoints")
trainer.fit(model, train_dataloaders=dataloader)