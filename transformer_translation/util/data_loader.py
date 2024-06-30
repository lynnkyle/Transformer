import torch
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import Multi30k


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, path, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.path = path
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(init_token=self.init_token, eos_token=self.eos_token, lower=True,
                                tokenize=self.tokenize_de, batch_first=True)
            self.target = Field(init_token=self.init_token, eos_token=self.eos_token, lower=True,
                                tokenize=self.tokenize_en, batch_first=True)
        elif self.ext == ('.en', '.de'):
            self.source = Field(init_token=self.init_token, eos_token=self.eos_token, lower=True,
                                tokenize=self.tokenize_en, batch_first=True)
            self.target = Field(init_token=self.init_token, eos_token=self.eos_token, lower=True,
                                tokenize=self.tokenize_de, batch_first=True)
        train_data, test_data, val_data = Multi30k.splits(path=self.path, exts=self.ext,
                                                          fields=(self.source, self.target),
                                                          train='train', validation='val', test='test')
        return train_data, test_data, val_data

    def make_iter(self, train_data, test_data, val_data, batch_size, device):
        train_iter, test_iter, val_iter = BucketIterator.splits((train_data, test_data, val_data),
                                                                batch_size=batch_size, device=device)
        return train_iter, test_iter, val_iter


from transformer_translation.util.tokenizer import Tokenizer

if __name__ == '__main__':
    tokenizer = Tokenizer()
    loader = DataLoader(ext=('.en', '.de'), tokenize_en=tokenizer.tokenize_en, tokenize_de=tokenizer.tokenize_de,
                        init_token='<sos>', eos_token='<eos>')
    train_data, test_data, val_data = loader.make_dataset()
    print("训练集===》", train_data, "测试集===》", test_data, "验证集===》", val_data)
    device = torch.device("cuda:0")
    train_iter, test_iter, val_iter = loader.make_iter(train_data, test_data, val_data, batch_size=128, device=device)
    print("训练集迭代器===》", train_iter, "测试集迭代器===》", test_iter, "验证集迭代器===》", val_iter)
