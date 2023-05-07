import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.functional import accuracy


class Transformer(pl.LightningModule):
    def __init__(self, input_num, output_num, feature_num):
        super().__init__()

        self.input_num = input_num
        self.output_num = output_num
        self.feature_num = feature_num

        self.scaler = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_num*self.feature_num, out_features=1)
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_num, nhead=1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=4)

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_num*self.feature_num, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4),
        )

        self.loss = nn.L1Loss()
    
    def forward(self, x):
        bs = x.shape[0]
        scales = self.scaler(x)
        x = self.encoder(torch.reshape(torch.add(scales.expand(bs, self.feature_num * self.input_num), torch.reshape(x, (bs, -1))), (bs, self.input_num, self.feature_num)))
        return torch.add(self.out(x), scales)
    
    def configure_optimizers(self):
        params = self.parameters()
        optimizer = torch.optim.SGD(params, lr=0.0001)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
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