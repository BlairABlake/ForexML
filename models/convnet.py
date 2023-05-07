from datasets import ForexPricePredictionDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

class Convnet(nn.Module):
    def __init__(self, input_num=32):
        super().__init__()

        self.input_num = input_num
        
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding="same"),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding="same"),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding="same"),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=input_num//8, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )
    
    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_num = 32
    model = Convnet(input_num).double().to(device=device)
    dataset = ForexPricePredictionDataset("./data/USDJPY_H1.csv", header=0, data_order="c", input_duration=input_num, output_duration=1, normalize=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    epochs = 1000

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(dataloader):
            x, y = data
            x = torch.squeeze(x, 2)
            x = torch.unsqueeze(x, 1)
            optimizer.zero_grad()
            outputs = model(x.to(device=device))
            loss = criterion(outputs, y.to(device=device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999: 
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
                running_loss = 0.0
    
    torch.save(model.state_dict(), "./model_states")
    
    model = Convnet(input_num=input_num).double()
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
