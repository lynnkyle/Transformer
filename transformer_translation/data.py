import conf
from util.tokenizer import Tokenizer
from util.data_loader import DataLoader

tokenizer = Tokenizer()

loader = DataLoader(path=conf.dataset_path, ext=('.en', '.de'), tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de, init_token='<sos>',
                    eos_token='<eos>')

train_data, test_data, val_data = loader.make_dataset()
loader.build_vocab(train_data, min_freq=conf.min_freq)
train_iter, test_iter, val_iter = loader.make_iter(train_data, test_data, val_data, batch_size=conf.batch_size,
                                                   device=conf.device)

src_pad_idx = loader.source.vocab.stoi['<pad>']
tgt_pad_idx = loader.target.vocab.stoi['<pad>']
tgt_sos_idx = loader.target.vocab.stoi['<sos>']

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)

if __name__ == '__main__':
    print("src_pad_idx===>", src_pad_idx)
    print("trg_pad_idx===>", tgt_pad_idx)
    print("trg_sos_idx===>", tgt_sos_idx)
    print("enc_vocab_size", enc_voc_size)
    print("dec_vocab_size", dec_voc_size)
