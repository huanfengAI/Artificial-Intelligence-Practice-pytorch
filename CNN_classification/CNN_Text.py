from collections import defaultdict
import time
import random
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class CNN_Text(nn.Module):
    
    def __init__(self,embed_num,embed_dim,class_num,Ci,kernel_num,kernel_sizes):
        super(CNN_Text, self).__init__()
        self.embed = nn.Embedding(embed_num, embed_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, class_num)

    def forward(self, x):
        #对于batch=1的x，shape=[words]
        x = self.embed(x)  # shape=[words,embed_dim]
        x=x.unsqueeze(0)#[1,words,embed_dim]
        x = x.unsqueeze(1)  # [1,1,words,embed_dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [1,kernel_num,(words-kernel_sizes)/2+1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [1,kernel_num]
        x = torch.cat(x, 1)#[1,kernel_num*len(kernel_sizes)]
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit



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
WIN_SIZE=[3,4,5]
FILTER_SIZE=8

#embed_num,embed_dim,class_num,Ci,kernel_num,kernel_sizes
model=CNN_Text(nwords,EMB_SIZE,ntags,1,FILTER_SIZE,WIN_SIZE)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())

for epoch in range(100):
    train_loss=0.0
    for _,words,tag in train:
        if len(words)<WIN_SIZE[2]:
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
        if len(words)<WIN_SIZE[2]:
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

