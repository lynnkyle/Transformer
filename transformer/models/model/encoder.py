from collections import OrderedDict

import torch
from torch import nn
from models.embedding.transformer_embedding import TransformerEmbedding
from models.blocks.encoder_layer import EncoderLayer
from util.test import seed_torch


class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, d_model, max_len, drop_prob, device, n_head, ffn_hidden, n_layers=6):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(enc_vocab_size, d_model, max_len, drop_prob, device)
        self.n_encoder = nn.ModuleList([EncoderLayer(d_model, n_head, ffn_hidden, drop_prob) for i in range(n_layers)])

    def forward(self, x, mask):
        out = self.embedding(x)
        for enc in self.n_encoder:
            out = enc(out, mask)


if __name__ == '__main__':
    seed_torch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    x = torch.randint(0, 5, size=(7, 5)).to(device)
    encoder = Encoder(enc_vocab_size=5, d_model=8, max_len=10, drop_prob=0.1, device=device, n_head=2,
                      ffn_hidden=2048).to(device)
    print(encoder(x, None))
