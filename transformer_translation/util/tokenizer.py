import spacy


class Tokenizer():
    def __init__(self):
        self.spacy_en = spacy.load("en_core_web_sm")
        self.spacy_de = spacy.load("de_core_news_sm")

    def tokenize_en(self, text):
        return [token.text for token in self.spacy_en.tokenizer(text)]

    def tokenize_de(self, text):
        return [token.text for token in self.spacy_de.tokenizer(text)]


if __name__ == '__main__':
    tokenizer = Tokenizer()
    text = "Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt."
    a = tokenizer.spacy_de(text)
    b = tokenizer.spacy_de.tokenizer(text)
    print(a, b)
