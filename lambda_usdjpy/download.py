import torch
model = torch.jit.load("time2vec.pt")
model.eval()