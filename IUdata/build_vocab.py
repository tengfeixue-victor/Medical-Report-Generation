# from github: https://github.com/ZexinYan/Medical-Report-Generation
import pickle
from collections import Counter
import json


class JsonReader(object):
    def __init__(self, json_file, generate_pkl=False):
        self.data = self.__read_json(json_file)
        self.gen_pkl = generate_pkl

        # for generating the pkl file
        if self.gen_pkl:
            self.keys = list(self.data.keys())

        self.items = list(self.data.items())

    def __read_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, item):
        # for generating the pkl file
        if self.gen_pkl:
            data = self.data[self.keys[item]]
        else:
            data = self.items[item]
        return data

    def __len__(self):
        return len(self.data)


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def get_word_by_id(self, id):
        return self.id2word[id]

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json_file, threshold, generate_pkl=False):
    caption_reader = JsonReader(json_file, generate_pkl)
    counter = Counter()

    for items in caption_reader:
        # The replace operation is used to keep the "," and "."
        text = items[0].replace('.', ' . ').replace(',', ' , ') + ' ' + items[1].replace('.', ' . ').replace(',', ' , ')
        counter.update(text.lower().split(' '))
    words = [word for word, cnt in counter.items() if cnt > threshold and word != '']
    vocab = Vocabulary()

    for word in words:
        print(word)
        vocab.add_word(word)
    return vocab


def main(json_file, threshold, vocab_path, generate_pkl=False):
    vocab = build_vocab(json_file, threshold, generate_pkl)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size:{}".format(len(vocab)))
    print("Saved path in {}".format(vocab_path))


"""Generate .pkl dictionary from .json data (impression and findings)"""
if __name__ == '__main__':
    # threshold is the appearance frequency of words
    generate_pkl = True
    json_file = 'IUdata_trainval.json'
    threshold = 0
    vocab_path = 'IUdata_vocab.pkl'
    main(json_file, threshold, vocab_path, generate_pkl)
