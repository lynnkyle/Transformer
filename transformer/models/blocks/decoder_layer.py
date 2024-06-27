import torch
from torch import nn
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.layer_norm import LayerNorm
from models.layers.position_wise_feed_forward import PositionwiseFeedForward
from util.test import seed_torch


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, ffn_hidden=2048, drop_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec_x, enc_x, tgt_mask, src_mask):
        _x = dec_x
        out = self.self_attention(q=dec_x, k=dec_x, v=dec_x, mask=tgt_mask)
        out = self.dropout1(out)
        out = self.norm1(out + _x)
        _x = out
        if enc_x is not None:
            out = self.cross_attention(q=out, k=enc_x, v=enc_x, mask=src_mask)
            out = self.dropout2(out)
            out = self.norm2(out + _x)
            _x = out
        out = self.ffn(out)
        out = self.dropout3(out)
        out = self.norm3(out + _x)
        return out


if __name__ == '__main__':
    seed_torch()
    enc_x = torch.randn(size=(1, 2, 16))
    dec_x = torch.randn(size=(1, 2, 16))
    encoderLayer = DecoderLayer(d_model=16, n_head=2)
    print(encoderLayer(enc_x, dec_x, src_mask=None, tgt_mask=None))
