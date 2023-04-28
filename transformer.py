from datasets import ForexDataWithInterval, ForexPricePredictionDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Transformer(nn.Module):
    def __init__(self, input_num, output_num, feature_num):
        super().__init__()

        self.input_num = input_num
        self.output_num = output_num
        self.feature_num = feature_num
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_num, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=4)

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_num*self.feature_num, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4),
        )

        self.f_i = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_num*self.feature_num, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4),
        )
    
    def forward(self, x):
        s = torch.sum(x)
        x_f = self.encoder(torch.div(x, s))
        x_f = self.out(x_f)
        x_i = self.f_i(x)
        return torch.add(x_f, x_i)


    
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_num = 20
    output_num = 1
    batch_size = 128
    model = Transformer(input_num,output_num, 4).double().to(device=device)
    dataset = ForexPricePredictionDataset("./data/USDJPY_H1.csv", header=0, data_order="ohlc", input_duration=input_num, output_duration=output_num)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    epochs = 100
    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in tqdm(enumerate(dataloader)):
            x, y = data
            optimizer.zero_grad()
            outputs = model(
                x.to(device=device)
            )
            loss = criterion(outputs, y.to(device=device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
        running_loss = 0.0
    
    torch.save(model.state_dict(), "./model_states")
    
    model = Transformer(input_num=input_num).double()
    model.load_state_dict(torch.load("./model_states"))
    ticker = yf.Ticker("USDJPY=X")
    data = ticker.history(period='40h', interval="1h")
    data = data["Close"].to_numpy()
    with torch.no_grad():
        data = torch.tensor(data[-input_num:])
        prediction = model(data).numpy()[0]
        data = np.append(data, prediction)

    plt.plot(data)
    plt.show()
