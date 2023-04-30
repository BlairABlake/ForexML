from torch import nn
from torchvision.transforms import Normalize, Resize, ToTensor, Compose, ToPILImage
import torch.nn as nn
import torch
import numpy as np
import yfinance as yf
import mplfinance as mpf
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import json
from datetime import timedelta  

model = torch.jit.load("time2vec.pt")

def data2image(data):
    fig,ax = mpf.plot(data, type='candle', returnfig=True, scale_padding=0, style='charles')
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

def predict(data, model):
    image = data2image(data)
    image = preprocessing(image)

    with torch.no_grad():
        pred = model(torch.unsqueeze(image, dim=0))
        prob = nn.functional.softmax(pred, dim=1).numpy()[0]
    
    return prob

def prob2label(prob):
  return np.random.choice(["down", "stationary", "up"], p=prob)

def most_probable(prob):
    return ["down", "stationary", "up"][np.argmax(prob)]
        

def lambda_handler(event, context):
    payload = event.get("payload")
    if payload is not None and payload.get("data") is not None:
        data = payload.get("data")
    print(payload)

    return json.dumps({
    })