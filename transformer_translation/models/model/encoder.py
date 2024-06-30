import torch
from torch import nn
from transformer_translation.models.embedding.transformer_embedding import TransformerEmbedding
from transformer_translation.models.blocks.encoder_layer import EncoderLayer
from transformer_translation.script.test import seed_torch


class Encoder(nn.Module):
    def __init__(self, enc_voc_size, d_model, max_len, drop_prob, device, n_head, ffn_hidden, n_layers=6):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, drop_prob, device)
        self.n_encoder = nn.ModuleList([EncoderLayer(d_model, n_head, ffn_hidden, drop_prob) for _ in range(n_layers)])

    def forward(self, enc_x, mask):
        out = self.embedding(enc_x)
        for layer in self.n_encoder:
            out = layer(out, mask)
        return out


if __name__ == '__main__':
    seed_torch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    x = torch.randint(0, 5, size=(7, 5)).to(device)
    encoder = Encoder(enc_voc_size=5, d_model=8, max_len=10, drop_prob=0.1, device=device, n_head=2,
                      ffn_hidden=2048).to(device)
    print(encoder(x, None))
