import torch

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)


import torch
def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')
device = get_device()   