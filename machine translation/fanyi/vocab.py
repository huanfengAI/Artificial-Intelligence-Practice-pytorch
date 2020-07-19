import pickle
class Src_vob(object):
    def __init__(self):
        self.word2idx={}
        self.idx2word={}
        self.idx=0
    def add_word(self,word):
        if word not in self.word2idx:
            self.word2idx[word]=self.idx
            self.idx2word[self.idx]=word
            self.idx+=1
    def __call__(self,word):
       if word not in self.word2idx:
           return self.word2idx['<unk>']
       return self.word2idx[word]
   
    def __len__(self):
       return len(self.word2idx)

class Tag_vob(object):
    def __init__(self):
        self.word2idx={}
        self.idx2word={}
        self.idx=0
    def add_word(self,word):
        if word not in self.word2idx:
            self.word2idx[word]=self.idx
            self.idx2word[self.idx]=word
            self.idx+=1
    def __call__(self,word):
       if word not in self.word2idx:
           return self.word2idx['<unk>']
       return self.word2idx[word]
   
    def __len__(self):
       return len(self.word2idx)


src_vob=Src_vob()
src_vob.add_word('<pad>')
src_vob.add_word('<s>')
src_vob.add_word('</s>')
src_vob.add_word('<unk>')


tag_vob=Tag_vob()
tag_vob.add_word('<pad>')
tag_vob.add_word('<s>')
tag_vob.add_word('</s>')
tag_vob.add_word('<unk>')

def read_tagfile(filename):
    with open(filename,"r")  as f:
        for line in f:
            words=line.strip().split()
            for word in words:
                tag_vob.add_word(word)

def read_srcfile(filename):
    with open(filename,"r")  as f:
        for line in f:
            words=line.strip().split()
            for word in words:
                src_vob.add_word(word)

read_srcfile("./data/train.ja")
read_tagfile("./data/train.en")
print(len(src_vob))
print(len(tag_vob))
#for k,v in tag_vob.word2idx.items():
#    print(k,v)
with open('src_vob.pkl','wb') as f:
    pickle.dump(src_vob,f,pickle.HIGHEST_PROTOCOL)
with open('tag_vob.pkl','wb') as f:
    pickle.dump(tag_vob,f,pickle.HIGHEST_PROTOCOL)
