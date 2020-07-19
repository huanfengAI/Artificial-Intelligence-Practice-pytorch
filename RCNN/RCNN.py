import torch
from torch import nn
import random
from torch.nn import functional as F
from collections import defaultdict,Counter
import nltk
import numpy as np
import pickle
from nltk.corpus import stopwords
import random
import csv


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


#方式一
class RCNN(nn.Module):
    def __init__(self,nwords,emb_size,hidden,hiddens_size_fc,nlayers,ntags):
        super(RCNN,self).__init__()
        #print(hidden)
        self.emb=nn.Embedding(nwords,emb_size)
        nn.init.xavier_uniform_(self.emb.weight)
        self.rnn=nn.LSTM(input_size=emb_size,hidden_size=hidden,num_layers=nlayers,dropout=0.5,bidirectional = True)
        self.linear=nn.Linear(2*hidden+emb_size,hiddens_size_fc)
        self.tanh=nn.Tanh()
        self.fc = nn.Linear(hiddens_size_fc,ntags)
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        x=self.emb(x)
        new_x=x.unsqueeze(1)#添加维度[seq_len, batch_size, embed_size]
        x,_=self.rnn(new_x)#[seq_len, batch_size, hidden_size*2]
        input=torch.cat([x,new_x],2).permute(1,0,2)#[batch,seq_len,emb_size+2*hidden_sizes]
        input=self.linear(input)#[batch,seq_len,hiddens_size_fc]
        input_output=self.tanh(input).permute(0,2,1)#[batch,hiddens_size_fc,seq_len]
        max_out_features = F.max_pool1d(input_output, input_output.shape[2]).squeeze(2)#[batch,hiddens_size_fc]
        max_out_features = self.dropout(max_out_features)
        out = self.fc(max_out_features)
        return out
#方式二
class RCNN1(nn.Module):
    def __init__(self,nwords,emb_size,hidden,hiddens_size_fc,nlayers,ntags):
        super(RCNN1,self).__init__()
        #print(hidden)
        self.emb=nn.Embedding(nwords,emb_size)
        self.rnn=nn.LSTM(input_size=emb_size,hidden_size=hidden,num_layers=nlayers,dropout=0.5,bidirectional = True)
        self.linear=nn.Linear(2*hidden+emb_size,hiddens_size_fc)
        self.tanh=nn.Tanh()
        self.fc = nn.Linear(hiddens_size_fc,ntags)
        #self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        x=self.emb(x)
        new_x=x.unsqueeze(1)#添加维度[seq_len, batch_size, embed_size]
        x,_=self.rnn(new_x)#[seq_len, batch_size, hidden_size*2]
        input=torch.cat([x,new_x],2)#[seq_len,batch,emb_size+2*hidden_sizes]
        input=self.linear(input)#[seq_len,batch,hiddens_size_fc]
        input_output=self.tanh(input).permute(1,2,0)#[batch,hiddens_size_fc,seq_len]
        max_out_features = F.max_pool1d(input_output, input_output.shape[2]).squeeze(2)#[batch,hiddens_size_fc]
        #max_out_features = self.dropout(max_out_features)
        out = self.fc(max_out_features)
        return out
model=RCNN(nwords=len(w2i),emb_size=64,hidden=128,hiddens_size_fc=32,nlayers=2,ntags=2)
#model1=RCNN1(nwords=len(w2i),emb_size=64,hidden=128,hiddens_size_fc=32,nlayers=2,ntags=2)



criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())
for epoch in range(8):
    train_loss=0.0
    model.train()
    random.shuffle(data)
    for words,tag in data:
        words_tensor=torch.tensor(words)
        tag_tensor=torch.tensor([tag])
        scores=model(words_tensor)
        loss=criterion(scores,tag_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    print("epoch=%r,loss=%.4f"%(epoch,train_loss/len(data)))

model.eval()   
answer=[]
for words in test:
        words_tensor=torch.tensor(words)
        scores=model(words_tensor)
        predict=scores.detach().argmax().item()
        #print(predict)
        answer.append(predict)
lines = []

for idx,value in enumerate(answer):
        line = '%s,%s\n'%(idx+1,value)
        lines.append(line)

with open('key6.csv', 'w') as f:
        f.writelines(lines)
