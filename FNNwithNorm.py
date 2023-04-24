from datasets import ForexPricePredictionDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class FNNwithNorm(nn.Module):
    def __init__(self, input_num):
        super().__init__()

        self.input_num = input_num
        self.model_f = nn.Sequential(
            nn.Linear(in_features=input_num, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
            nn.Tanh()
        )

        self.model_i = nn.Sequential(
            nn.Linear(in_features=input_num, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        s = torch.sum(x)
        x_f = self.model_f(torch.div(x, s))
        x_i = self.model_i(x)
        return torch.add(x_f, x_i)

input_num = 20
model = FNNwithNorm(input_num).double()
dataset = ForexPricePredictionDataset("./data/USDJPY_H1.csv", header=0, data_order="c", input_duration=input_num, output_duration=1, normalize=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()

epochs = 100

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
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
            running_loss = 0.0

