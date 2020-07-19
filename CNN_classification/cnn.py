from collections import defaultdict
import time
import random
import torch
from torch import nn
import numpy as np


class MY_NET(nn.Module):
    def __init__(self,nwords,emb_size,ntags,wid_size,num_filters):
        super(MY_NET,self).__init__()
        self.emb=nn.Embedding(nwords,emb_size)
        nn.init.uniform_(self.emb.weight,-0.25,0.25)
        self.conv_1d=nn.Conv1d(emb_size,num_filters,wid_size)
        self.relu=nn.ReLU()
        self.out=nn.Linear(num_filters,ntags)
        nn.init.uniform_(self.out.weight,-0.25,0.25)
    def forward(self,x):#一位卷积需要[batch，通道数，长度]
        emb=self.emb(x)
        emb=emb.unsqueeze(0).permute(0,2,1)
        h=self.conv_1d(emb)
        h=h.max(dim=2)[0]
        h=self.relu(h)
        features=h.squeeze(0)#emb_size
        out=self.out(h)
        return out

w2i=defaultdict(lambda :len(w2i))
UNK=w2i["<unk>"]
t2i=defaultdict(lambda :len(t2i))
def read_dataset(filename):
    with open(filename,"r") as f:
        for line in f:
            tag,words=line.lower().strip().split(" ||| ")
            yield(words,[w2i[word] for word in words.split(" ")],t2i[tag])

train=list(read_dataset("train.txt"))
w2i=defaultdict(lambda : UNK,w2i)
dev=list(read_dataset("dev.txt"))
nwords=len(w2i)
ntags=len(t2i)
print(nwords)
print(ntags)

EMB_SIZE=64
WIN_SIZE=3
FILTER_SIZE=8


model=MY_NET(nwords=nwords,emb_size=EMB_SIZE,ntags=ntags,wid_size=WIN_SIZE,num_filters=FILTER_SIZE)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())

for epoch in range(100):
    train_loss=0.0
    for _,words,tag in train:
        if len(words)<WIN_SIZE:
            words+=[0]*(WIN_SIZE-len(words))
        words_tensor=torch.tensor(words)
        tag_tensor=torch.tensor([tag])
        scores=model(words_tensor)
        loss=criterion(scores,tag_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    print("epoch=%r,loss=%.4f"%(epoch,train_loss/len(train)))

    model.eval()
    dev_loss,dev_correct=0.0,0.0
    for _,words,tag in dev:
        if len(words)<WIN_SIZE:
            words+=[0]*(WIN_SIZE-len(words))
        words_tensor=torch.tensor(words)
        tag_tensor=torch.tensor([tag])
        scores=model(words_tensor)
        loss=criterion(scores,tag_tensor)
        dev_loss+=loss
        predict=scores.detach().argmax().item()
        if predict==tag:
            dev_correct+=1
    print("epoch=%r,loss=%.4f,acc=%.4f"%(epoch,dev_loss/len(dev),dev_correct/len(dev)))

