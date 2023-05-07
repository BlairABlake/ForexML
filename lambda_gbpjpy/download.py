import torch
model = torch.jit.load("time2vec_gbpjpy.pt")
model.eval()