import torch
from torch import nn
import random
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        #embedded = [src sent len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [src sent len, batch size, hid dim]
        # hidden, cell: [n layers* n directions, batch size, hid dim]

       
        return hidden, cell

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [1, batch size, hid dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        #output.squeeze(0)=[batch size,hid dim]
        prediction = self.out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell

# seq2seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
      
       
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)

        # 编码器的最后一个隐藏状态用作解码器的初始隐藏状态
        hidden, cell = self.encoder(src)

       
        input = trg[0, :]#torch.Size([64])
        #max_len包含最终一个</s>,所以并不需要预测</s>的后面的是什么
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            #print(output.shape)#torch.Size([64, 7044])
            
            #将预测放在一个张量中，其中包含每个标记的预测
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            #最有可能的值
            top1 = output.argmax(1)

            #要么在下一时刻输入预测的，要么输入真实的
            input = trg[t] if teacher_force else top1

        return outputs

