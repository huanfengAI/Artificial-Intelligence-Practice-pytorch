import torch
from torch.utils.data import Dataset
def readfile(src,target):
    with open(src,"r") as f_src,open(target,"r") as f_trg:
        for line_src,line_trg in zip(f_src,f_trg):
            sent_src=[x for x in line_src.strip().split()]
            sent_trg=[x for x in line_trg.strip().split()]
            yield(sent_src,sent_trg)
class TextDataset(Dataset):
    def __init__(self,src,target,src_vocab,tag_vocab,dataload=readfile):
        '''
        这里只是为了读取文件，然后封装成一个样本的模样
        '''
        data=list(dataload(src,target))
        self.data=data
        self.length=len(data)
        self.src_vocab=src_vocab
        self.tag_vocab=tag_vocab
    def __getitem__(self,idx):
        tokens=self.data[idx]
        src_vocab=self.src_vocab
        tag_vocab= self.tag_vocab
        src_caption=[]
        tag_caption=[]
        src_caption.append(src_vocab('<s>'))
        src_caption.extend([src_vocab(token) for token in tokens[0]])
        src_caption.append(src_vocab('</s>'))
        
       
        tag_caption.append(tag_vocab('<s>'))
        tag_caption.extend([tag_vocab(token) for token in tokens[1]])
        tag_caption.append(tag_vocab('</s>'))
       
        src=torch.LongTensor(src_caption)
        tag=torch.LongTensor(tag_caption)
        return src,tag
    def __len__(self):
        return self.length
        
        
def collate_fn(data):
    # Sort a data list by caption length
  
    src_captions,tag_captions = zip(*data)
    src_lengths = [len(src) for src in src_captions]
    tag_lengths = [len(tag) for tag in tag_captions]
    src=torch.zeros(len(src_captions),max(src_lengths)).long()
    tag=torch.zeros(len(tag_captions),max(tag_lengths)).long()
    for i,cap in enumerate(src_captions):
        end=src_lengths[i]
        src[i,:end]=cap[:end]
    for i,cap in enumerate(tag_captions):
        end=tag_lengths[i]
        tag[i,:end]=cap[:end]
    return src,tag
