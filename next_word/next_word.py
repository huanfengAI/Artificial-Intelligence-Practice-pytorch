import torch 
from torch import nn
from collections import defaultdict
import math
import random
class My_net(nn.Module):
    def __init__(self,nwords,emb_size,hid_size,num_hist):
        super(My_net,self).__init__()
        self.embedding=nn.Embedding(nwords,emb_size)
        self.fnn=nn.Sequential(
	    nn.Linear(num_hist*emb_size,hid_size),
            nn.Dropout(True),
            nn.Linear(hid_size,nwords)
	)
    def forward(self,x):
        emb=self.embedding(x)
        feat=emb.view(emb.size(0),-1)
        logit=self.fnn(feat)
        return logit
N=2#两个单词预测下一个词
EMB_SIZE=128
HID_SIZE=128
MAZ_LEN=50
w2i=defaultdict(lambda:len(w2i))
S=w2i["<s>"]#既做开始又做结束
UNK=w2i["<unk>"]
MAX_LEN=100
def generate_sent():
    hist=[S]*N
    sent=[]
    while True:
        new_hist=torch.LongTensor([hist])
        logits=model(new_hist)
        prob = nn.functional.softmax(logits)
        next_word = prob.multinomial(1).data[0,0]
        if next_word == S or len(sent) == MAX_LEN:
            break
        sent.append(next_word)
        hist = hist[1:] + [next_word]
    return sent  
def read_dataset(filename):
    with open(filename,"r") as f:
        for line in f:
            yield[w2i[x] for x in line.strip().split(" ")]
train=list(read_dataset("datalab/39811/train.txt"))
w2i=defaultdict(lambda : UNK,w2i)
dev=list(read_dataset("datalab/39811/valid.txt"))
i2w={v:k for k,v in w2i.items()}
nwords=len(w2i)
model=My_net(nwords=nwords,emb_size=EMB_SIZE,hid_size=HID_SIZE,num_hist=N)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()
for epoch in range(10):
    random.shuffle(train)
    model.train()
    train_words,train_loss=0,0.0
    for sent_id,sent in enumerate(train):#遍历每一条文本
        hist=[S]*N#[0,0]
        all_histories=[]#一条文本的所有特征[[x1,x2],[x2,x3],..]
        all_targets=[]#一条文本的所有的标签[y1,y2,y3...]
        for next_word in sent+[S]:
            all_histories.append(hist)
            all_targets.append(next_word)
            hist=hist[1:]+[next_word]
        all_histories=torch.LongTensor(all_histories)
        tag=torch.tensor(all_targets)
        logits=model(all_histories)
        loss = criterion(logits,tag)
        train_loss+=loss.item()
        train_words+=len(sent)#样本的个数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch %s ,loss %.4f"%(epoch,train_loss/train_words))
    model.eval()
    dev_words,dev_loss=0,0.0
    for sent_id,sent in enumerate(dev):
        hist=[S]*N
        all_histories=[]
        all_targets=[]
        for next_word in sent+[S]:
            all_histories.append(list(hist))
            all_targets.append(next_word)
            hist=hist[1:]+[next_word]
        all_histories=torch.LongTensor(all_histories)
        tag=torch.tensor(all_targets)
        logits=model(all_histories)
        loss=criterion(logits,tag)
        dev_loss+=loss.item()
        dev_words+=len(sent)
    print("devepoch %s ,loss %.4f"%(epoch,dev_loss/dev_words))
        
    for _ in range(5):
        sent=generate_sent()
        print(" ".join([i2w[x.item()] for x in sent]))     