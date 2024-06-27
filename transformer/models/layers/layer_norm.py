import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        # print(mean, var)
        x = (x - mean) / torch.sqrt(var + self.eps)
        y = self.gamma * x + self.beta
        return y


if __name__ == '__main__':
    layerNorm = LayerNorm(5)
    tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
    print(layerNorm(tensor))
