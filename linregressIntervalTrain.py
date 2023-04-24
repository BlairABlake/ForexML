from datasets import ForexDataLinregresswithInterval
import torch.nn as nn
from torch.utils.data import DataLoader
import torch

class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=16, kernel_size=10, stride=1, padding="same"),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=1, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1)
        )
    
    def forward(self, x):
        return self.model(x)

model = Net().double()
dataset = ForexDataLinregresswithInterval("./data/USDJPY_H1.csv", header=0, interval=20)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()

epochs = 10000

for epoch in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        x, y = data
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999: 
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
