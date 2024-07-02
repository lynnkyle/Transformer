import torch
from torch import nn

from models.model.encoder import Encoder
from models.model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, tgt_pad_idx, tgt_sos_idx, enc_voc_size, dec_voc_size, d_model, max_len, drop_prob,
                 device, n_head, ffn_hidden, n_layers):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.encoder = Encoder(enc_voc_size, d_model, max_len, drop_prob, device, n_head, ffn_hidden, n_layers)
        self.decoder = Decoder(dec_voc_size, d_model, max_len, drop_prob, device, n_head, ffn_hidden, n_layers)
        self.device = device

    def forward(self, enc_x, dec_x):
        src_mask = self.make_src_mask(enc_x)
        tgt_mask = self.make_tgt_mask(dec_x, self.device)
        out = self.encoder(enc_x, src_mask)
        out = self.decoder(dec_x, out, tgt_mask, src_mask)
        return out

    # enc_x shape: [batch_size,seq_len]
    # src_mask shape: [batch_size,n_head,seq_len,seq_len]
    def make_src_mask(self, enc_x):
        src_mask = (enc_x != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    # enc_x shape: [batch_size,seq_len]
    # src_mask shape: [batch_size,n_head,seq_len,seq_len]
    def make_tgt_mask(self, dec_x, device):
        seq_len = dec_x.shape[1]
        tgt_mask = (dec_x != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        tril_mask = torch.tril(torch.ones(seq_len, seq_len)).type(torch.BoolTensor).to(device)
        tgt_mask = tgt_mask & tril_mask
        return tgt_mask


from transformer_translation.script.test import seed_torch

if __name__ == '__main__':
    seed_torch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    trans = Transformer(src_pad_idx=1, tgt_pad_idx=1, tgt_sos_idx=2, enc_voc_size=10,
                        dec_voc_size=10, d_model=8, max_len=5, drop_prob=0.1, device=device, n_head=2, ffn_hidden=2048,
                        n_layers=6)
    trans.to(device)
    enc_x = torch.randint(0, 9, size=(7, 5)).to(device)
    print(enc_x)
    dec_x = torch.randint(0, 9, size=(7, 5)).to(device)
    print(dec_x)
    back = trans(enc_x, dec_x)
    print(back)
