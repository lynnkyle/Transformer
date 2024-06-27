from util.tokenizer import Tokenizer
from util.data_loader import DataLoader


def fun(**kwargs):
    print(kwargs.pop("am"))


if __name__ == '__main__':
    fun(I=1, am=2, kangkang=3)
