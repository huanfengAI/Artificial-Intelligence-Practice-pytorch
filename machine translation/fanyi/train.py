import time
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
import pickle
from vocab import Src_vob
from vocab import Tag_vob
from dataset import TextDataset
from dataset import collate_fn
from model.seq2seq import Decoder, Encoder,Seq2Seq
from evaluate import evaluate
BATCH_SIZE=64



src_vocab=pickle.load(open('src_vob.pkl','rb'))
tag_vocab=pickle.load(open('tag_vob.pkl','rb'))
train_dataset=TextDataset("./data/train.ja","./data/train.en",src_vocab,tag_vocab)
dev_dataset=TextDataset("./data/dev.ja","./data/dev.en",src_vocab,tag_vocab)
test_dataset=TextDataset("./data/test.ja","./data/test.en",src_vocab,tag_vocab)

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn,drop_last=True)
dev_loader=torch.utils.data.DataLoader(dataset=dev_dataset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate_fn,drop_last=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate_fn,drop_last=True)


INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(tag_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec)




# init weights
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

# calculate the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
optimizer = torch.optim.Adam(model.parameters())
PAD_IDX = tag_vocab('<pad>')
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)


def train(model,train_loader,optimizer,criterion,clip):
    model.train()
    epoch_loss=0
    for i,batch in enumerate(train_loader):
        src=batch[0].permute(1,0)
        trg=batch[1].permute(1,0)
        
        output=model(src,trg)
        #output的维度为[seq_len,batch,nwords]，#其中seq_len的第一个我们并没有放入预测的词所以
        #output[1:]
        output=output[1:].view(-1,output.shape[-1])
        trg=trg[1:].contiguous().view(-1)
        #将二者展开计算损失
        #print(output.shape)#torch.Size([1472, 7044])
        #print(trg.shape)#torch.Size([1472])
        loss=criterion(output,trg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss+=loss.item()
    return epoch_loss/len(train_loader)        
    
    
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model,dev_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #存储dev最好的模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')

model.load_state_dict(torch.load('tut1-model.pt'))
#验证
test_loss = evaluate(model,test_loader, criterion)

print(f'| Test Loss: {test_loss:.3f}')

