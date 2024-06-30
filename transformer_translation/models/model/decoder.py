import torch
from torch import nn
from transformer_translation.models.embedding.transformer_embedding import TransformerEmbedding
from transformer_translation.models.blocks.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, d_model, max_len, drop_prob, device, n_head, ffn_hidden, n_layers=6):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)
        self.n_decoder = nn.ModuleList([DecoderLayer(d_model, n_head, ffn_hidden, drop_prob) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec_x, enc_x, tgt_mask, src_mask):
        out = self.embedding(dec_x)
        for layer in self.n_decoder:
            out = layer(out, enc_x, tgt_mask, src_mask)
        out = self.linear(out)
        return out


import torch
from transformer_translation.script.test import seed_torch

if __name__ == '__main__':
    seed_torch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    enc_x = torch.randint(0, 5, size=(7, 5))
    ebed = nn.Embedding(5, 8)
    enc_x = ebed(enc_x).to(device)
    dec_x = torch.randint(0, 5, size=(7, 5)).to(device)
    decoder = Decoder(dec_voc_size=5, d_model=8, max_len=5, drop_prob=0.1, device=device, n_head=2,
                      ffn_hidden=2048, n_layers=6).to(device)

    print(decoder(dec_x, enc_x, None, None))
