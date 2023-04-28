from datasets import ForexPricePredictionDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

class FNNwithNorm(nn.Module):
    def __init__(self, input_num):
        super().__init__()

        self.input_num = input_num

        self.model_f = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )
        self.rnn_f = nn.LSTM(input_size=1, hidden_size=128)
        self.conv_f = nn.Sequential(
            nn.Conv1d(in_channels=input_num, out_channels=32, kernel_size=4, padding="same"),
            nn.MaxPool1d(4, 4),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, padding="same"),
            nn.MaxPool1d(4, 4),
            nn.ReLU()
        )
        self.model_i = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_num, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )
    
    def forward(self, x):
        s = torch.sum(x)
        x_f, _ = self.rnn_f(torch.div(x, s))
        x_f = self.conv_f(x_f)
        x_f = self.model_f(x_f)
        x_i = self.model_i(x)
        return torch.add(x_f, x_i)
    
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_num = 40
    model = FNNwithNorm(input_num).double().to(device=device)
    dataset = ForexPricePredictionDataset("./data/USDJPY_H1.csv", header=0, data_order="c", input_duration=input_num, output_duration=1, normalize=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    epochs = 1000

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(dataloader):
            x, y = data
            optimizer.zero_grad()
            outputs = model(
                torch.unsqueeze(x, 2).to(device=device)
            )
            loss = criterion(outputs, y.long().to(device=device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999: 
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
                running_loss = 0.0
    
    torch.save(model.state_dict(), "./model_states")
    
    model = FNNwithNorm(input_num=input_num).double()
    model.load_state_dict(torch.load("./model_states"))
    ticker = yf.Ticker("USDJPY=X")
    data = ticker.history(period='40h', interval="1h")
    data = data["Close"].to_numpy()
    for i in range(10):
        with torch.no_grad():
            data = torch.tensor(data[-input_num:])
            prediction = model(data).numpy()[0]
            data = np.append(data, prediction)

    plt.plot(data)
    plt.show()
