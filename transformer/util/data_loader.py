from torchtext.legacy.data import Field, BucketIterator


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en):
        self.ext = ext
