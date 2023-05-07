from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.datasets import ImageFolder
from torch import optim, nn, utils, Tensor
from torchvision.transforms import Normalize, Resize, ToTensor, Compose, ToPILImage
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from models.NTM import NTM
import json

transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_dataset = ImageFolder(
    "images/train",
    transforms,
)
test_dataset = ImageFolder(
    "images/test",
    transforms
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

memory_df = pd.read_csv("vecs.csv", names=["datetime", "vec"])
memory = memory_df["vec"].apply(lambda s: s.replace("[", "").replace("]", "").replace("\n", "").replace(" ", ",")).apply(lambda s: s.split(",")).apply(lambda xs: list(filter(lambda s: s != '', xs))).apply(lambda xs: np.array(list(map(float, xs)))).to_numpy()
memory = torch.from_numpy(np.stack(memory, axis=0))

model = NTM(memory, time2vec_model_path="./time2vec_usdjpy.ckpt")
trainer = pl.Trainer(max_epochs=10, enable_progress_bar=True, accelerator="cpu", default_root_dir="./checkpoints")
trainer.fit(model, train_dataloaders=train_loader)
trainer.test(model, test_loader)