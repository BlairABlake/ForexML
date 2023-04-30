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
    payload = event.get("payload")
    interval = event.get("interval")
    if payload is not None and payload.get("queryStringParameters") is not None:
        interval = payload["queryStringParameters"].get("interval")
    
    interval = "15m" if interval is None else interval
    ticker = yf.Ticker("USDJPY=X")
    data_now = ticker.history(interval=interval, period="1wk")
    data_now = data_now[-30:] if len(data_now.index) >= 30 else data_now
    down, stationary, up = predict(data_now, model)

    return json.dumps({
        "datetime": (data_now.iloc[-1].name + timedelta(hours=7)).strftime("%m/%d/%Y, %H:%M:%S"),
        "interval": interval,
        "up": str(up),
        "down": str(down),
        "stationary": str(stationary)
    })