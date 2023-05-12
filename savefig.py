import mplfinance as mpf
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os
from multiprocessing import Pool
from contextlib import closing

from datasets import ForexPricePredictionDataset

def savecandle(data, root, label, name):
    fig,ax = mpf.plot(data.head(30), type='candle', returnfig=True, scale_padding=0, style='charles')
    ax[0].set_axis_off()
    if label == 1:
        path = f"./{root}/up/"
    elif label == 2:
        path = f"./{root}/down/"
    else:
        path = f"./{root}/stationary/"
    fig.savefig(path + name + ".png", pad_inches=0)
    ax[0].cla()
    ax[1].cla()

def _save(d, root, duration, threshold):
    y = d.iloc[duration]
    x = d.iloc[:duration]

    if y["Close"] - x["Close"][-1] > threshold:
        label = 1
    elif y["Close"] - x["Close"][-1] < -threshold:
        label = 2
    else:
        label = 0
    savecandle(x, root, label, str(x.iloc[0].name))
    plt.figure().clear()
    plt.close('all')
    plt.cla()
    plt.clf()

def _save_wrapper(args):
   _save(*args)

def saveall(data, root, duration=30, interval=15, threshold=0.03, progress_bar=False):
    with closing(Pool()) as p:
        if progress_bar:
            list(tqdm(p.imap(_save_wrapper, [[data.iloc[i-(duration+1):i], root, duration, threshold] for i in range(duration+1, len(data), interval)]), total=len(data)-(duration+interval)))
        else:
            p.starmap(_save, [[data.iloc[i-(duration+1):i], root, duration, threshold] for i in range(duration+1, len(data), interval)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default="")
    parser.add_argument("-s", default="image")
    parser.add_argument("--interval", default=1, type=int)
    parser.add_argument("--threshold", default=0.03, type=float)
    parser.add_argument("--progress", type=bool, default=True)
    args = parser.parse_args()

    if args.f == "":
        print("Usage: -f path to the data file")
        exit()

    dataset = ForexPricePredictionDataset(args.f, header=0, data_order="tohlc", input_duration=30, output_duration=1, normalize=False)
    save = args.s
    try:
        os.mkdir(os.path.join(save, "up"))
        os.mkdir(os.path.join(save, "down"))
        os.mkdir(os.path.join(save, "stationary"))
    except:
        pass
    saveall(dataset.data, save, interval=args.interval, threshold=args.threshold, progress_bar=args.progress)
