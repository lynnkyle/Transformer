import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--int", action="append")
parser.add_argument("--test", action="append")
parser.add_argument("--use_vocab", default=False)

parser = parser.parse_args()
print(parser)
