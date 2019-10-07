import torch
from torch import nn
import random
import numpy as np
from collections import defaultdict

class DeepCbow(nn.Module):
    def __init__(self,n_words,ntags,nlayers,emb_size,hid_size):
        super(DeepCbow,self).__init__()
        self.nlayers = nlayers
        self.embedding=nn.Embedding(n_words,emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.linears = nn.ModuleList([nn.Linear(emb_size if i == 0 else hid_size, hid_size)   for i in range(nlayers)])
        for i in range(nlayers):
            nn.init.xavier_uniform_(self.linears[i].weight)
        self.out=nn.Linear(hid_size,ntags)
        nn.init.xavier_uniform_(self.out.weight)
    def forward(self,words):
        emb=self.embedding(words)
        emb_sum=torch.sum(emb,dim=0)
        h=emb_sum.view(1,-1)
        for i in range(self.nlayers):
            h=self.linears[i](h)
            h=torch.tanh(h)
        out=self.out(h)
        return out

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])


train = list(read_dataset("train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("test.txt"))
nwords = len(w2i)
ntags = len(t2i)
EMB_SIZE = 64
HID_SIZE = 64
NLAYERS = 2
model = DeepCbow(nwords, ntags, NLAYERS, EMB_SIZE, HID_SIZE)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())


for epoch  in range(100):
    random.shuffle(train)
    train_loss=0.0
    for words,tag in train:
        words=torch.tensor(words)
        tag=torch.tensor([tag])
        score=model(words)
        loss=criterion(score,tag)
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("current %r,currernt loss %.4f"%(epoch,train_loss/len(train)))
   
    test_correct=0.0
    for words,tag in dev:
        words=torch.tensor(words)
        scores=model(words).detach().numpy()
        predict=np.argmax(scores)
        if predict==tag:
            test_correct+=1
    print("current epoc %r test acc=%.4f"%(epoch,test_correct/len(dev)))
