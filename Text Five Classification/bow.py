import torch
from torch import nn
from collections import defaultdict
import random
import numpy as np
class Bow(nn.Module):
    def __init__(self,nwords,ntags):
        super(Bow,self).__init__()
        self.bias=torch.zeros(ntags,requires_grad=True)
        self.embedding=nn.Embedding(nwords,ntags)
        nn.init.xavier_uniform_(self.embedding.weight)
    def forward(self,words):
        emb=self.embedding(words)
        out=torch.sum(emb,dim=0)+self.bias
        out=out.view(1,-1)
        return out

w2i=defaultdict(lambda:len(w2i))
t2i=defaultdict(lambda:len(t2i))
UNK=w2i["<unk>"]


def read_dataset(filename):
    with open(filename,'r') as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

train=list(read_dataset("train.txt"))

w2i = defaultdict(lambda: UNK, w2i)#
dev=list(read_dataset("test.txt"))
nwords=len(w2i)
ntags=len(t2i)


model=Bow(nwords,ntags)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())

for epoch in range(100):
    random.shuffle(train)
    train_loss=0.0
    for words,tag in train:
        words=torch.tensor(words)
        tag=torch.tensor([tag])
        print(tag.shape)
        score=model(words)
        loss=criterion(score,tag)
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch%20==0:
         print(train_loss/len(train))

    test_correct=0.0
    for words,tag in dev:
        words=torch.tensor(words)
        scores=model(words).detach().numpy()
        predict=np.argmax(scores)
        if predict==tag:
            test_correct+=1
    print("current epoc %r test acc=%.4f"%(epoch,test_correct/len(dev)))
