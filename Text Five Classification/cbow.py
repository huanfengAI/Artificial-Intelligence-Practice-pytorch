import torch
from torch import nn
from collections import defaultdict
import numpy as np
import random
class Cbow(nn.Module):
    def __init__(self,nwords,ntags,emb_size):
        super(Cbow,self).__init__()
        self.embedding=nn.Embedding(nwords,emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.linear=nn.Linear(emb_size,ntags)
        nn.init.xavier_uniform_(self.linear.weight)
    def forward(self,words):
        emd=self.embedding(words)
        emb_sum=torch.sum(emd,dim=0)
        emd_sum=emb_sum.view(1,-1)
        out=self.linear(emd_sum)
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

EMB_SIZE=32
model=Cbow(nwords,ntags,EMB_SIZE)
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
