import torch
from torch import nn
import test

if __name__ == '__main__':
    x = torch.autograd.Variable(torch.Tensor([5]), requires_grad=True)
    y = torch.autograd.Variable(torch.Tensor([5]), requires_grad=True)
    z = y + x
    z.backward()
    print(x.grad)
    torch.nn.utils.clip_grad_norm_(parameters=(x, y), max_norm=1)
    print(x.grad)
