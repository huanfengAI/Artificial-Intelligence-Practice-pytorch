from torch.autograd import Variable
import numpy as np
import torch
from model.seq2seq import EncoderRNN,AttnDecoderRNN

import matplotlib.pyplot as plt

# 从一个列表到句子
def SentenceFromList(tag_vocab, lst):
    result = [tag_vocab.idx2word[int(i)] for i in lst if i != 3]
    
    result = ' '.join(result)
    return(result)
def indexesFromSentence(lang, sentence):
    return [lang.word2idx[word] for word in sentence.split(' ')]
    
    
def indexFromSentence(lang, sentence,MAX_LENGTH):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(2)
    for i in range(MAX_LENGTH - len(indexes)):
        indexes.append(EOS_token)
    return(indexes)


def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素
    
    
def test(test_dataset,encoder,decoder,tag_vocab,src_tag):
    indices = np.random.choice(range(len(test_dataset)), 20)
    for ind in indices:
        data=test_dataset[ind]
        src=data[0].view(1,-1)[:,0:10]
        trg=data[1].view(1,-1)[:,0:10]
        
        max_length=trg.size(1)#记录当前样本的长度
        encoder_hidden = encoder.initHidden(src.size()[0])
        encoder_outputs, encoder_hidden = encoder(src, encoder_hidden)
        decoder_input = torch.LongTensor([[1]] * trg.size()[0])
        decoder_hidden = encoder_hidden
        output_sentence = []#记录每个时刻的预测
        print(max_length)
        decoder_attentions = torch.zeros(max_length, max_length)#记录attention的矩阵
        rights = []
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
            print(decoder_attention.shape)
            #print(decoder_attention.squeeze(0).squeeze(0).data)
            #print(decoder_attention.data.shape)
            print(decoder_attentions.shape)
            print(decoder_attention.squeeze(0).data.shape)
            if(decoder_attention.size(2)>max_length):
                decoder_attention=decoder_attention[:,:,0:max_length]
            print(decoder_attention.shape)
            decoder_attentions[di] = decoder_attention.squeeze(0).data
            topv, topi = decoder_output.data.topk(1, dim = 1)
            ni = topi[:, 0]
            decoder_input = ni.unsqueeze(1)
            ni = ni.numpy()[0]
            output_sentence.append(ni)
            # decoder_input大小：batch_size, length_seq
            print(di)
            right = rightness(decoder_output, trg[:, di])
            #right为正确的
            rights.append(right)
        sentence = SentenceFromList(tag_vocab, output_sentence)
        standard = SentenceFromList(tag_vocab, trg[0])
        print('机器翻译：', sentence)
        print('标准翻译：', standard)
        # 输出本句话的准确率
        right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
        print('词准确率：', 100.0 * right_ratio)
        print('\n')
    
    '''
    rights = []
    # 通过几个特殊的句子翻译，考察注意力机制关注的情况
    input_sentence = '件 の 一 件 で メール を いただ き ありがとう ござ い ま し た 。'
    data = np.array([indexFromSentence(src_tag, input_sentence)])
    input_variable = Variable(torch.LongTensor(data))[:,0:10]
    # input_variable的大小：batch_size, length_seq
    target_variable = Variable(torch.LongTensor(trg))
    # target_variable的大小：batch_size, length_seq

    encoder_hidden = encoder.initHidden(input_variable.size()[0])

    loss = 0
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    # encoder_outputs的大小：batch_size, length_seq, hidden_size*direction
    # encoder_hidden的大小：direction*n_layer, batch_size, hidden_size

    decoder_input = Variable(torch.LongTensor([[1]] * target_variable.size()[0]))
    # decoder_input大小：batch_size, length_seq
    decoder_input = decoder_input

    decoder_hidden = encoder_hidden
    # decoder_hidden大小：direction*n_layer, batch_size, hidden_size

    output_sentence = []
    decoder_attentions = torch.zeros(max_length, max_length)
    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_outputs)
        #decoder_ouput大小：batch_size, output_size(vocab_size)
        topv, topi = decoder_output.data.topk(1, dim = 1)
    
        # 在每一步，获取了注意力的权重向量，并将其存储到了decoder_attentions之中
        decoder_attentions[di] = decoder_attention.data
        #topi 尺寸：batch_size, k
        ni = topi[:, 0]
        decoder_input = Variable(ni.unsqueeze(1))
        ni = ni.numpy()[0]
        output_sentence.append(ni)
        # decoder_input大小：batch_size, length_seq
        decoder_input = decoder_input
        right = rightness(decoder_output, target_variable[:, di])
        rights.append(right)
    sentence = SentenceFromList(tag_vocab, output_sentence)
    print('机器翻译：', sentence)
    print('\n')
    # 将每一步存储的注意力权重组合到一起就形成了注意力矩阵，绘制为图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(decoder_attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # 设置坐标轴
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                   ['</s>'], rotation=90)
    ax.set_yticklabels([''] + sentence.split(' '))

    # 在标度上展示单词
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    '''
