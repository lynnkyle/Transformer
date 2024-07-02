import math
import time

from data import *
from torch import nn, optim
from models.model.transformer import Transformer
from script.test import seed_torch
from transformer_translation.util.epoch_timer import get_time_diff

seed_torch()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)
    # if hasattr(m, 'bias'):
    #     nn.init.normal_(m.bias.data)


model = Transformer(src_pad_idx=src_pad_idx,
                    tgt_pad_idx=tgt_pad_idx,
                    tgt_sos_idx=tgt_sos_idx,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    d_model=d_model,
                    max_len=max_len,
                    drop_prob=drop_prob,
                    device=device,
                    n_head=n_heads,
                    ffn_hidden=ffn_hidden,
                    n_layers=n_layers).to(device)

model.apply(initialize_weights)

optimizer = optim.Adam(model.parameters(),
                       lr=init_lr,
                       eps=adam_eps,
                       weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, train_iter, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(train_iter):
        src = batch.src
        tgt = batch.trg
        optimizer.zero_grad()
        out = model(src, tgt[:, :-1])
        out_reshape = out.contiguous().view(-1, out.shape[-1])
        target = tgt[:, 1:].contiguous().view(-1)
        loss = criterion(out_reshape, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        print('step:', round(i * 100 / len(train_iter), 2), "%", "loss:", loss.item())
    return epoch_loss / len(train_iter)


def evaluate(model, val_iter, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad:
        for i, batch in enumerate(val_iter):
            src = batch.src
            tgt = batch.trg
            optimizer.zero_grad()
            out = model(src, tgt[:, :-1])
            out_reshape = out.contiguous().view(-1, out.shape[-1])
            target = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(out_reshape, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
    return epoch_loss / len(val_iter)


def run(total_epoch, best_loss):
    for epoch in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss = evaluate(model, val_iter, criterion)
        print("第{}轮迭代的损失值:".format(epoch), valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'save/transformer-{}.pth'.format(epoch))
        print(f'Epoch: {epoch + 1} | Time: ', get_time_diff(start_time))
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
