import torch
from torch import nn
from collections import defaultdict,Counter
import torch.utils.data as data
import nltk
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
import pickle
from nltk.corpus import stopwords
import random
import csv
import torch.nn.functional as F
#nltk.download('stopwords')


nltk.download('stopwords')
w2i=defaultdict(lambda:len(w2i))
t2i=defaultdict(lambda:len(t2i))
HAM=t2i["ham"]
spam=t2i["spam"]
UNK=w2i["<unk>"]

def read_data(filename):
    with open(filename,"r") as f:
        for line in f:
            #print(line.strip().lower().split("\t"))
            tag,words=line.strip().lower().split("\t")
            yield([w2i[word] for word in words.split(" ") if word not in stopwords.words('english')],t2i[tag])
                
data=list(read_data("sms_spam/sms_train.txt"))
nwords=len(w2i)
ntags=len(t2i)
emb_size=128
WIN_SIZE=3
num_filters=8
def read_test_data(filename):
    with open(filename,"r") as f:
        for line in f:
            words=line.strip().lower()
            yield[w2i[word] for word in words.split(" ") if word not in stopwords.words('english')]
w2i=defaultdict(lambda:UNK,w2i)            
test=list(read_test_data("sms_spam/sms_test.txt"))

class Model(nn.Module):
    def __init__(self,nwords,emb_size,nlayers,hidden_size,hidden_size2,ntags):
        super(Model, self).__init__()
        self.embedding=nn.Embedding(nwords,emb_size)
        self.lstm=nn.LSTM(emb_size,hidden_size,nlayers,bidirectional=True,dropout=0.5,batch_first=True)
        self.tanh1=nn.Tanh()
        self.w=nn.Parameter(torch.Tensor(hidden_size*2))
        self.fc1=nn.Linear(hidden_size*2,hidden_size2)
        self.fc2=nn.Linear(hidden_size2,ntags)
    def forward(self, x):
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        emb=emb.unsqueeze(0)#[batch_size, seq_len, embeding]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]
        M = self.tanh1(H)  # [128, 32, 256][batch_size, seq_len, hidden_size * num_direction]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out) 
        out = self.fc2(out)  # [128, 64]
        return out
        
        
class MY_NET(nn.Module):
    def __init__(self,nwords,emb_size,nlayers,hidden_size,hidden_size2,ntags):
        super(MY_NET,self).__init__()
        self.embedding=nn.Embedding(nwords,emb_size)
        self.lstm=nn.LSTM(emb_size,hidden_size,nlayers,bidirectional=True,dropout=0.5)
        self.tanh1=nn.Tanh()
        self.w=nn.Parameter(torch.Tensor(hidden_size*2))
        self.fc=nn.Linear(hidden_size*2,hidden_size2)
        self.out=nn.Linear(hidden_size2,ntags)
    def forward(self,x):
        emb=self.embedding(x)#[seq_len,emb_size]
        emb=emb.unsqueeze(1)#[seq_len,1,emb_size]
        H,_=self.lstm(emb)#[seq_len,1,hidden_size*2]
        M=self.tanh1(H)#[seq_len,1,hidden_size*2]
        #print(self.w.shape)[hidden_size*2]
        #print(torch.matmul(M, self.w).shape)[seq_len,1]
        #print(F.softmax(torch.matmul(M, self.w), dim=0).shape)#[seq_len,1]
        alpha = F.softmax(torch.matmul(M, self.w), dim=0).unsqueeze(-1)#[seq_len,1,1]
        out=H*alpha#[seq_len,1,hidden_size*2]
        out=torch.sum(out,0)#所有时间维度求和[1,hidden_size*2]
        out=F.relu(out)#[1,hidden_size*2]
        out=self.fc(out)#[1,hidden_size2]
        out=self.out(out)#[hidden_size2,ntags]
        return out



model=MY_NET(nwords,emb_size,2,128,64,ntags)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())

for epoch in range(10):
    data_loss=0.0
    model.train()
    random.shuffle(data)
    for words,tag in data:
        words_tensor=torch.tensor(words)
        tag_tensor=torch.tensor([tag])
        scores=model(words_tensor)
        #print(scores.shape)
        loss=criterion(scores,tag_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        data_loss+=loss.item()
    print("epoch=%r,loss=%.4f"%(epoch,data_loss/len(data)))

model.eval()
answer=[]
for words in test:
        if len(words)<WIN_SIZE:
            words+=[0]*(WIN_SIZE-len(words))
        words_tensor=torch.tensor(words)
        scores=model(words_tensor)
        predict=scores.detach().argmax().item()
        #print(predict)
        answer.append(predict)
lines = []

for idx,value in enumerate(answer):
        line = '%s,%s\n'%(idx+1,value)
        lines.append(line)

with open('key2.csv', 'w') as f:
        f.writelines(lines)
torch.save(model.state_dict(), 'attention1.pt')    
