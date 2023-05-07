import torch
model = torch.jit.load("time2vec_audjpy.pt")
model.eval()