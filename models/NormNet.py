import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.functional import accuracy


class NormNet(pl.LightningModule):
    def __init__(self, input_num, output_num):
        super().__init__()

        self.input_num = input_num
        self.output_num = output_num
        self.hidden_size =128

        self.scaler = nn.Sequential(
            nn.BatchNorm1d(num_features=self.input_num),
            nn.Flatten(),
            nn.Linear(in_features=self.input_num, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1)
        )

        self.rnn = nn.Sequential(
            nn.BatchNorm1d(num_features=self.input_num),
            nn.LSTM(input_size=1, num_layers=8, hidden_size=self.hidden_size, batch_first=True)
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_num * self.hidden_size, out_features=self.output_num)
        )
        

        self.loss = nn.MSELoss()
    
    def forward(self, x):
        bs = x.shape[0]
        scales = self.scaler(x)
        scaled = torch.add(x, torch.unsqueeze(scales.expand(bs, self.input_num), 2))
        rnn_out, _ = self.rnn(scaled)
        return torch.add(self.out(rnn_out), scales)

    
    def configure_optimizers(self):
        params = self.parameters()
        optimizer = torch.optim.SGD(params, lr=0.0001)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, torch.squeeze(targets, 2))
        self.log("train_loss", loss, prog_bar=True)
        return {
            "loss": loss
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
    