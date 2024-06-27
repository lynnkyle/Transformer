import torch
from torch import nn
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward
from util.test import seed_torch


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, ffn_hidden=2048, drop_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask):
        out = self.attention(q=x, k=x, v=x, mask=mask)
        out = self.dropout1(out)
        out = self.norm1(out + x)
        x = out
        out = self.ffn(out)
        out = self.dropout2(out)
        out = self.norm2(out + x)
        return out


if __name__ == '__main__':
    seed_torch()
    x = torch.randn(size=(1, 2, 16))
    encoderLayer = EncoderLayer(d_model=16, n_head=2)
    print(encoderLayer(x, None))
