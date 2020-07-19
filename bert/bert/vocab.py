import os
import json
l=1
word2idx={}
pad_index = 0 # 用来补长度和空白
unk_index = 1 # 用来表达未知的字, 如果字典里查不到
cls_index = 2 #CLS#
sep_index = 3 #SEP#
mask_index = 4 # 用来做Masked LM所做的遮罩
num_index = 5 # (可选) 用来替换语句里的所有数字, 例如把 "23.9" 直接替换成 #num#,这里并没有换
word2idx["#PAD#"] = pad_index
word2idx["#UNK#"] = unk_index
word2idx["#SEP#"] = sep_index
word2idx["#CLS#"] = cls_index
word2idx["#MASK#"] = mask_index
word2idx["#NUM#"] = num_index

def totalword2idx(word):
    if word not in word2idx:
        word2idx[word]=len(word2idx)

def word_char(text):
    wordlist=list(text)
    for word in wordlist:
        totalword2idx(word)
with open("./pretraining_data/wiki_dataset/train_wiki1.txt", "r", encoding="ISO-8859-1") as f:
    for line in f:
        try:
             line=line.encode("ISO-8859-1").decode("UTF-8")
             if l==1:
                 line=eval(line)
                 text1=line["text1"].strip()
                 text2=line["text2"].strip()
                 word_char(text1)
                 word_char(text2)
                 l=l+1
             else:
                 line=eval(line)
                 text=line["text2"].strip()
                 word_char(text)
                 if l%10000==0:
                     print(l)
                 l=l+1
        except:
            pass
            
with open('./pretraining_data/wiki_dataset/bert_word2idx1.json', 'w+', encoding='utf-8') as f:
    f.write(json.dumps(word2idx, ensure_ascii=False))

        

       

    

