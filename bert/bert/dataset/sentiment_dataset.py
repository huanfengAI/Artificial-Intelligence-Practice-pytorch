from torch.utils.data import Dataset
from sklearn.utils import shuffle
import random
import re
import torch
class CLSDataset(Dataset):
    def __init__(self,corpus_path,word2idx,max_seq_len,data_regularization=False):
        self.corpus_path=corpus_path
        self.word2idx=word2idx#字典对象
        self.max_seq_len=max_seq_len
        self.data_regularization=data_regularization
        self.pad_index=0
        self.unk_index=1
        self.cls_index=2
        self.sep_index=3
        self.mask_index=4
        self.num_index=5
        with open(corpus_path,"r",encoding="utf-8") as f:
            self.lines=[eval(line) for line in f]
            self.lines=shuffle(self.lines)
            self.length=len(self.lines)
    def __getitem__(self, item):
        line=self.lines[item]
        text=line["text"]
        label=line["label"]
        if self.data_regularization:
            #数据正则，有10%的几率截取句子的一部分
            if random.random()<0.1:
                split_spans=[i.span() for i in re.finditer(",|;|。|？|！",text)]
                if len(split_spans)!=0:
                    span_idx=random.randint(0,len(split_spans)-1)
                    cut_position=split_spans[span_idx][1]
                    if random.random()<0.5:
                        if len(text)-cut_position>2:
                            text=text[cut_position]
                        else:
                            text=text[:cut_position]
                    else:
                        if cut_position>2:
                            text=text[:cut_position]
                        else:
                            text=text[cut_position:]
        text=self.token_char(text)
        text_input=[self.cls_index]+text+[self.sep_index]
        text_input = text_input[:self.max_seq_len]
        output={"text_input": torch.tensor(text_input),
                "label":torch.tensor([label])}
        return output

    def token_char(self,text):
        return [self.word2idx.get(word,self.unk_index) for word in text]




    def __len__(self):
        return self.length
