import lightning.pytorch as pl
from torchmetrics.functional import accuracy
import torch.nn as nn
import torch
import models.time2vec as time2vec
import numpy as np

class ReadingLayer(nn.Module):
    def forward(self, memory, weight):
        weight = nn.functional.softmax(weight, dim=1)
        return torch.from_numpy(np.array([torch.matmul(torch.transpose(memory, 0, 1), w.double()).detach().numpy() for w in weight]))

class Vec2Weight(nn.Module):
    def __init__(self, vec_length, memory_size):
        super().__init__()
        self.linear = nn.Linear(in_features=vec_length, out_features=memory_size)

    def forward(self, vec):
        return self.linear(vec)
    
class NTM(pl.LightningModule):
    def __init__(self, memory, out_num=3, time2vec_model_path=""):
        super().__init__()

        self.memory = memory
        memory_size, vec_length = memory.shape

        self.time2vec_model = time2vec.Time2Vec.load_from_checkpoint(time2vec_model_path, map_location="cpu")

        self.reading = ReadingLayer()
        self.vec2weight = Vec2Weight(vec_length, memory_size)

        self.out = nn.Sequential(
            nn.Linear(in_features=vec_length, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=out_num),
            nn.Softmax(dim=1)
        ).double()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, inp):
        vec = time2vec.time2vec(inp, self.time2vec_model)
        w = self.vec2weight(vec)
        x = self.reading(self.memory, w)
        return self.out(x)
    
    def configure_optimizers(self):
        params = self.parameters()
        optimizer = torch.optim.SGD(params, lr=0.0001)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        train_accuracy = accuracy(outputs, targets, "multiclass", num_classes=3)
        loss = self.loss(outputs, targets)
        self.log("train_accuracy", train_accuracy, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return {
            "loss": loss,
            "train_accuracy": train_accuracy
        }

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        test_accuracy = accuracy(outputs, targets, "multiclass", num_classes=3)
        loss = self.loss(outputs, targets)
        self.log("test_accuracy", test_accuracy)
        return {
            "test_loss": loss,
            "test_accuracy": test_accuracy
        }
