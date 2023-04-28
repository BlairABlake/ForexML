from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.datasets import ImageFolder
from torch import optim, nn, utils, Tensor
from torchvision.transforms import Normalize, Resize, ToTensor, Compose, ToPILImage
from torchvision.models import resnet50, ResNet50_Weights
import lightning.pytorch as pl
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import yfinance as yf
import mplfinance as mpf
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

class Time2Vec(pl.LightningModule):
    def __init__(self, vec_length=100):
        super().__init__()

        self.vec_length = vec_length

        self.resnet = resnet50(ResNet50_Weights)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc_vec = nn.Linear(in_features=1000, out_features=self.vec_length)
        self.out = nn.Linear(in_features=self.vec_length, out_features=3)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc_vec(x)
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

def data2image(data):
    fig,ax = mpf.plot(data[-30:], type='candle', returnfig=True, scale_padding=0, style='charles')
    ax[0].set_axis_off()

    canvas = FigureCanvas(fig)
    canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    ax[0].cla()
    ax[1].cla()
    plt.figure().clear()
    plt.close('all')
    plt.cla()
    plt.clf()

    return image

def preprocessing(image):
    transforms = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return transforms(image)

def predict(data, model=None, checkpoint_path=""):
    image = data2image(data)
    image = preprocessing(image)
    if model is None:
        model = Time2Vec.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location="cpu")
    model.eval()

    with torch.no_grad():
        prediction = model(torch.unsqueeze(image, dim=0))
    
    return prediction

def pred2label(pred):
  prob = nn.functional.softmax(pred, dim=1).numpy()[0]
  return np.random.choice(["down", "stationary", "up"], p=prob)



def time2vec(data):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    model = Time2Vec()
    model.fc_vec.register_forward_hook(get_activation('fc_vec'))
    prediction = predict(data, model)
    return activation["fc_vec"]
        

if __name__ == "__main__":
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

    model = Time2Vec()
    trainer = pl.Trainer(max_epochs=10, enable_progress_bar=True, accelerator="cpu", default_root_dir="./checkpoints")
    trainer.fit(model, train_dataloaders=train_loader)
    trainer.test(model, test_loader)
