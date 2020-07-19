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
from model.seq2seq import EncoderRNN,AttnDecoderRNN
from evaluate import evaluate
import torch.nn.functional as F
import random
from plot_attention import test


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


'''
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec)
'''

hidden_size = 32
MAX_LENGTH = 10
n_layers = 1
encoder = EncoderRNN(INPUT_DIM, hidden_size, n_layers = n_layers)
decoder = AttnDecoderRNN(hidden_size, OUTPUT_DIM, dropout_p=0.5,max_length = MAX_LENGTH, n_layers= n_layers)
    
    
learning_rate = 0.0001
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

criterion = nn.NLLLoss()
teacher_forcing_ratio = 0.5


def train(encoder,decoder,train_loader,encoder_optimizer,decoder_optimizer,criterion):
    decoder.train()
    print_loss_total = 0
    for i,batch in enumerate(train_loader):
        src=batch[0][:,0:10]
        trg=batch[1][:,0:10]
        #print(src.shape)[64,10]
        #print(trg.shape)[64,10]
        #清空梯度
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        #src.size(0)为batch
        encoder_hidden = encoder.initHidden(src.size(0))
        #print(encoder_hidden.shape)#torch.Size([2, 64, 32])
        loss = 0

        #编码器开始工作
        encoder_outputs, encoder_hidden = encoder(src, encoder_hidden)
        
        #设置decoder的输入为[1],也就是<s>
        decoder_input = torch.LongTensor([[1]] * trg.size(0))
        #print(decoder_input.shape)#torch.Size([64, 1])
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # 用监督信息作为下一时刻解码器的输入
            # 开始时间不得循环
            for di in range(MAX_LENGTH):
                # 输入给解码器的信息包括输入的单词decoder_input, 解码器上一时刻的因曾单元状态，
                # 编码器各个时间步的输出结果
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                #decoder_ouput大小：batch_size, output_size
                #计算损失函数，得到下一时刻的解码器的输入
                loss += criterion(decoder_output, trg[:, di])
                decoder_input = trg[:, di].unsqueeze(1)  # Teacher forcing
                # decoder_input大小：batch_size, length_seq
        else:
            # 没有教师监督，用解码器自己的预测作为下一时刻的输入

            # 对时间步进行循环
            for di in range(MAX_LENGTH):
                #decoder_input表示decoder的输入
                #decoder_hidden=encider_hidden表示的decoder的最后一个时刻的输输出
                #encoder_outputs表示decoder端的所有时刻的输出
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                #decoder_ouput大小：batch_size, output_size(vocab_size)
                # 获取解码器的预测结果，并用它来作为下一时刻的输入
                topv, topi = decoder_output.data.topk(1, dim = 1)
                #topi 尺寸：batch_size, k
                ni = topi[:, 0]
                decoder_input = ni.unsqueeze(1)
                # decoder_input大小：batch_size, length_seq

                # 计算损失函数,这个谁每一步都计算损失
                loss += criterion(decoder_output, trg[:, di])
        loss.backward()
        
        encoder_optimizer.step()
        decoder_optimizer.step()
        print_loss_total += loss.data.numpy()

    print_loss_avg = print_loss_total / len(train_loader)
    return print_loss_avg       
    
    
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 30
CLIP = 1

best_valid_loss = float('inf')
plot_losses = []
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(encoder,decoder,train_loader,encoder_optimizer,decoder_optimizer,criterion)
    valid_loss,right_ratio = evaluate(encoder,decoder,dev_loader,criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #存储dev最好的模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(encoder.state_dict(), 'encoder.pt')
        torch.save(decoder.state_dict(), 'decoder.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
    print(f'\t Val. acc: {right_ratio:.3f}')

    plot_losses.append([train_loss, valid_loss / len(dev_loader), right_ratio])
a = [i[0] for i in plot_losses]
b = [i[1] for i in plot_losses]
c = [i[2] * 100 for i in plot_losses]
plt.plot(a, '-', label = 'Training Loss')
plt.plot(b, ':', label = 'Validation Loss')
plt.plot(c, '.', label = 'Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss & Accuracy')
plt.legend() 
plt.show()



encoder.load_state_dict(torch.load('encoder.pt'))
decoder.load_state_dict(torch.load('decoder.pt'))
#验证
test_loss,test_acc = evaluate(encoder,decoder,test_loader, criterion)
print("test_loss=%.4f,loss=%.4f"%(test_loss,test_acc))
#绘制注意力图像
#test(test_dataset,encoder,decoder,tag_vocab,src_vocab)
