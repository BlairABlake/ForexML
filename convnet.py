import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from datasets import ForexPricePredictionDataset

class ConvNet(nn.Module):
    def __init__(self, input_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=32,
                kernel_size=5,
                padding='same'
            ),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
if __name__ == "__main__":
    input_duraion = 20
    output_duration = 20
    lr = 1e-4
    num_epochs = 40

    dataset = ForexPricePredictionDataset(data_file="./data/USDJPY_H1.csv", input_duration=input_duraion, output_duration=output_duration, data_order="c", header=0)
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    traindata_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = ConvNet(input_size=input_duraion, output_size=output_duration)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for x, t in dataset_loader:
            optimizer.zero_grad()
            y = model(x.float())
            loss = loss_fn(y, t.float())
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        
        print(f"Epoch: {epoch}  Loss: {running_loss}")