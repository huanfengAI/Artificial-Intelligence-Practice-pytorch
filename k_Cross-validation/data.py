import jieba
'''
本方法是将本数据集中的数据中的文本数据放到email.txt文件中，然后将样本对应的标签放到labels.txt
'''

def tokenizer_jieba(line):
    return [li for li in jieba.cut(line) if li.strip() != '']



def read_data(filename):
    with open(filename,"r") as f:
        for line in f:
            #tag是标签，word是文本
            tag,words=line.strip().lower().split("\t")
  
            label_list = []
            if line[0] == 's':
                label_list.append('1')
            elif line[0] == 'h':
                label_list.append('0')   
            f = open("labels.txt", 'a+', encoding='utf8')
            f.write(' '.join(label_list)+'\n')
            f.close()
            f = open("email.txt", 'a+', encoding='utf8')
            email = [word for word in jieba.cut(words) if word.strip() != '']
            f.write(' '.join(email) + '\n')
            
read_data("sms_spam/sms_train.txt")
