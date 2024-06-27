import torch
from torch import nn
from util.test import seed_torch


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(-2, -1)
        atten_score = q @ k_t / torch.sqrt(torch.tensor(d_tensor))
        if mask is not None:
            atten_score = torch.masked_fill(atten_score, mask)
        atten_score = self.softmax(atten_score)
        v = atten_score @ v
        return v, atten_score

