import torch
from torch import nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True, 
                          num_layers = self.n_layers, bidirectional = True)

    def forward(self, input, hidden):
        #input： batch_size, length_seq
        embedded = self.embedding(input)
        #embedded：batch_size, length_seq, hidden_size
        output = embedded
        output, hidden = self.gru(output, hidden)
        # output：batch_size, length_seq, hidden_size
        # hidden：num_layers * directions, batch_size, hidden_size
        return output, hidden

    def initHidden(self, batch_size):
        # 对隐含单元变量全部进行初始化
        #num_layers * num_directions, batch, hidden_size
        result = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)
        return result
        
        
        
# 定义基于注意力的解码器RNN
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,max_length, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 词嵌入层
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 注意力网络（一个前馈神经网络）
        self.attn = nn.Linear(self.hidden_size * (2 * n_layers + 1), self.max_length)
        # 注意力机制作用完后的结果映射到后面的层
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        # dropout操作层
        self.dropout = nn.Dropout(self.dropout_p)
        # 定义一个双向GRU，并设置batch_first为True以方便操作
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, bidirectional = True,
                         num_layers = self.n_layers, batch_first = True)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)# embedded大小：batch_size, length_seq, hidden_size
        embedded = embedded[:, 0, :]# embedded大小：batch_size, hidden_size
        embedded = self.dropout(embedded)
        # 将hidden张量数据转化成batch_size排在第0维的形状
        temp_for_transpose = torch.transpose(hidden, 0, 1).contiguous()
        #temp_for_transpose大小：batch_size, direction*n_layer, hidden_size
        temp_for_transpose = temp_for_transpose.view(temp_for_transpose.size()[0], -1)
        #temp_for_transpose.shape大小：batch_size, direction*n_layer×hidden_size
        hidden_attn = temp_for_transpose
        input_to_attention = torch.cat((embedded, hidden_attn), 1)
        # input_to_attention大小：batch_size, hidden_size+ direction * n_layers×hidden_size
        attn_weights = F.softmax(self.attn(input_to_attention),dim=1) # 注意力层输出的权重
        # attn_weights大小：batch_size, max_length
        attn_weights = attn_weights[:, : encoder_outputs.size()[1]]
        attn_weights = attn_weights.unsqueeze(1)
        # 将attention的weights矩阵乘encoder_outputs以计算注意力完的结果
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        # attn_weights大小：batch_size, 1, seq_length 中间的1是为了bmm乘法用的 
        # encoder_outputs大小：batch_size, seq_length, hidden_size*direction
        # attn_applied大小：batch_size, 1, hidden_size*direction
        # bmm: 两个矩阵相乘。忽略第一个batch纬度，缩并时间维度
        # 将输入的词向量与注意力机制作用后的结果拼接成一个大的输入向量
        output = torch.cat((embedded, attn_applied[:,0,:]), 1)
        # output大小：batch_size, hidden_size * (direction + 1)
        output = self.attn_combine(output).unsqueeze(1)
        # output大小：batch_size, length_seq, hidden_size
        output = F.relu(output)
        output = self.dropout(output)
        output, hidden = self.gru(output, hidden)
        # output大小：batch_size, length_seq, hidden_size * directions
        # hidden大小：n_layers * directions, batch_size, hidden_size
        #取出GRU运算最后一步的结果喂给最后一层全链接层
        output = self.out(output[:, -1, :])
        # output大小：batch_size * output_size
        # 取logsoftmax，计算输出结果
        output = F.log_softmax(output, dim = 1)
        # output大小：batch_size * output_size
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        # 初始化解码器隐单元，尺寸为n_layers * directions, batch_size, hidden_size
        result = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)
        return result

