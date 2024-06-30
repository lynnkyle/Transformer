import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.device = device
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = torch.unsqueeze(pos, 0).transpose(0, 1)
        _2i = torch.arange(0, d_model, 2, device=device)
        _2i = torch.unsqueeze(_2i, 0)
        self.encoding[:, 0::2] = torch.sin(pos / torch.pow(10000, _2i / d_model))
        self.encoding[:, 1::2] = torch.cos(pos / torch.pow(10000, _2i / d_model))

    def forward(self, x):
        # batch_size, seq_len = x.shape
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


if __name__ == '__main__':
    # print(torch.cuda.is_available())
    tensor = torch.randn(3, 32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    pos_embedding = PositionalEncoding(32, 512, device)
    # print(pos_embedding(tensor))
