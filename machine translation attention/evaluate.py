import torch
from torch.autograd import Variable
import numpy as np
MAX_LENGTH=10
#验证只计算损失
use_cuda=False
def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素
def evaluate(encoder,decoder,dev_loader,criterion):
    decoder.eval()
    decoder.eval()
    valid_loss=0.0
    rights = []
    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            src = batch[0][:,0:10]
            trg = batch[1][:,0:10]
            encoder_hidden = encoder.initHidden(batch[0].size()[0])
            loss = 0
        encoder_outputs, encoder_hidden = encoder(src, encoder_hidden)
        # encoder_outputs的大小：batch_size, length_seq, hidden_size*direction
        # encoder_hidden的大小：direction*n_layer, batch_size, hidden_size

        decoder_input = Variable(torch.LongTensor([[1]] * trg.size()[0]))
        # decoder_input大小：batch_size, length_seq
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden
        # decoder_hidden大小：direction*n_layer, batch_size, hidden_size

        # 开始每一步的预测
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #decoder_ouput大小：batch_size, output_size(vocab_size)
            topv, topi = decoder_output.data.topk(1, dim = 1)
            #topi 尺寸：batch_size, k
            ni = topi[:, 0]

            decoder_input = Variable(ni.unsqueeze(1))
            # decoder_input大小：batch_size, length_seq
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            right = rightness(decoder_output, trg[:, di])
            rights.append(right)
            loss += criterion(decoder_output, trg[:, di])
        loss = loss.cpu() if use_cuda else loss
        valid_loss += loss.data.numpy()
    valid_loss = valid_loss / len(dev_loader)
    # 计算平均损失、准确率等指标并打印输出
    right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])

    return   valid_loss,right_ratio
