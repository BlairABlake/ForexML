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
model.eval()

interval2period = {
    "1m": "1d",
    "5m": "1d",
    "15m": "1d",
    "30m": "5d",
    "1h":  "5d",
    "1d": "3mo",
    "1wk": "5mo"
}

def data2image(data):
    fig,ax = mpf.plot(data, type='candle', returnfig=True, scale_padding=0, style='charles')
    ax[0].set_axis_off()

    canvas = FigureCanvas(fig)
    canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()

    ax[0].cla()
    ax[1].cla()
    plt.figure().clear()
    plt.close('all')
    plt.cla()
    plt.clf()

    return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

def preprocessing(image):
    return Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])(image)

def predict(data, model):
    with torch.no_grad():
        return nn.functional.softmax(
            model(torch.unsqueeze(preprocessing(data2image(data)), dim=0)), dim=1
        ).numpy()[0]

def prob2label(prob):
  return np.random.choice(["down", "stationary", "up"], p=prob)

def most_probable(prob):
    return ["down", "stationary", "up"][np.argmax(prob)]
        

def lambda_handler(event, context):
    interval = "15m"
    if event.get("queryStringParameters") is not None and  event["queryStringParameters"].get("interval") is not None:
        interval = event["queryStringParameters"]["interval"]
        
    ticker = yf.Ticker("USDJPY=X")
    data_now = ticker.history(interval=interval, period=interval2period[interval])
    data_now = data_now[-min(30, len(data_now.index)):]
    down, stationary, up = predict(data_now, model)

    return json.dumps({
        "datetime": (data_now.iloc[-1].name + timedelta(hours=8)).strftime("%m/%d/%Y, %H:%M:%S"),
        "interval": interval,
        "up": str(up),
        "down": str(down),
        "stationary": str(stationary)
    })