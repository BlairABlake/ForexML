import torch
model = torch.jit.load("time2vec_eurjpy.pt")
model.eval()