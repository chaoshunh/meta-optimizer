import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_batch

class LayerNorm1D(nn.Module):

    def __init__(self, num_outputs, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_outputs))
        self.bias = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
  #      x_mean = x.mean(1, keepdim=True).expand_as(x)
 #       x_std = x.std(1, keepdim=True).expand_as(x)
#        x = (x-x_mean) / (x_std+self.eps)
        return x*self.weight.expand_as(x) + self.bias.expand_as(x)

if __name__ == '__main__':
    model = LayerNorm1D(10)
    print(model)
    x, y = get_batch(2)
    print(x)
    print(model(x))
