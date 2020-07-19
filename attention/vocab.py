import torch
from torch import nn
from collections import defaultdict,Counter
import torch.utils.data as data
import nltk
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
import pickle
from nltk.corpus import stopwords
nltk.download('stopwords')
class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
counter=Counter()
def total_count(filename):
    with open(filename,"r") as f:
        for line in f:
            tag,words=line.strip().lower().split("\t")
            #count.update接收一句话中分词之后的列表
            counter.update([word for word in words.split(" ") if word not in stopwords.words('english')])
             # Create a vocab wrapper and add some special tokens.

total_count("sms_spam/sms_train.txt")
words = [word for word, cnt in counter.items() if cnt >= 3]
vocab = Vocabulary()
vocab.add_word('<pad>')
vocab.add_word('<start>')
vocab.add_word('<end>')
vocab.add_word('<unk>')
print(len(words))
for i, word in enumerate(words):
    vocab.add_word(word)
print(len(vocab))
with open('vocabnew1.pkl','wb') as f:
    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
 

