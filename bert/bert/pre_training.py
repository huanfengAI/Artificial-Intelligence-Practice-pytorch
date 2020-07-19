import json
import torch
import numpy as np
from models.bert_model import *
from dataset.wiki_dataset import BERTDataset
from torch.utils.data import DataLoader
import os
import tqdm
config={}
config["word2idx_path"]="./pretraining_data/wiki_dataset/bert_word2idx_extend.json"
config["train_data"]="./pretraining_data/wiki_dataset/train_wiki.txt"
config["test_data"]="./pretraining_data/wiki_dataset/test_wiki.txt"
config["output_path"] = "./output_wiki_bert"
config["epoch"]=0
config["train_epoch"]=10
config["on_memory"]=True
config["batch_size"]=1
config["max_seq_len"]=222
config["vocab_size"]=32162
config["lr"]=2e-6
config["num_workers"]=2

class Pertrainer:
    def __init__(self,BertForPreTraining,vocab_size,max_seq_len,lr,batch_size):
        self.vocab_size=vocab_size
        self.max_seq_len=max_seq_len
        self.lr=lr
        self.batch_size=batch_size
        use_cuda=torch.cuda.is_available()
        self.device=torch.device("cuda:0" if use_cuda else "cpu")
        bertconfig=BertConfig(vocab_size=vocab_size)
        self.bert_model=BertForPreTraining(config=bertconfig)
        self.bert_model.to(self.device)
        train_dataset=BERTDataset(corpus_path=config["train_data"],
                                  word2idx_path=config["word2idx_path"],
                                  seq_len=self.max_seq_len,
                                  hidden_dim=bertconfig.hidden_size,
                                  on_memory=config["on_memory"])
        self.train_dataloader=DataLoader(train_dataset,batch_size=self.batch_size,
                                         num_workers=config["num_workers"],collate_fn=lambda x:x)
        test_dataset=BERTDataset(corpus_path=config["test_data"],
                                 word2idx_path=config["word2idx_path"],
                                 seq_len=self.max_seq_len,
                                 hidden_dim=bertconfig.hidden_size,
                                 on_memory=config["on_memory"])
        self.test_dataloader=DataLoader(test_dataset,batch_size=self.batch_size,
                                        num_workers=config["num_workers"],collate_fn=lambda x:x)
        #[max_seq_len,hidden_size]
        self.positional_enc=self.init_positional_encoding(hidden_dim=bertconfig.hidden_size,
                                                          max_seq_len=self.max_seq_len)
        self.positional_enc=torch.unsqueeze(self.positional_enc,dim=0)
        #[1,max_seq_len,hidden_size]
        optim_paramers=list(self.bert_model.parameters())
        self.optimizer=torch.optim.Adam(optim_paramers,lr=self.lr)
        #p.nelement()可以统计出tensor中张量的个数
        print("Total Parameters:" ,sum([p.nelement() for p in self.bert_model.parameters()]))


    def init_positional_encoding(self,hidden_dim,max_seq_len):#这些都是常数
        position_enc=np.array([
            [pos/np.power(10000,2*i/hidden_dim) for i in range(hidden_dim)]
            if pos !=0  else np.zeros(hidden_dim) for pos in range(max_seq_len)
        ])
        position_enc[1:,0::2]=np.sin(position_enc[1:,0::2])
        position_enc[1:,1::2]=np.cos(position_enc[1:,1::2])
        denominator=np.sqrt(np.sum(position_enc**2,axis=1,keepdims=True))
        position_enc=position_enc/(denominator+1e-8)
        position_enc=torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc
    def load_model(self,model,dir_path):
        checkpoint_dir=self.find_most_recent_state_dict(dir_path)
        checkpoint=torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded for training!".format(checkpoint_dir))

    #寻找最新的文件
    def fing_most_recent_state_dict(self,dir_path):
        dic_list=[i for i in os.listdir(dir_path)]
        if len(dic_list)==0:
            raise FileNotFoundError("no model file")
        dic_list=[i for i in dic_list if "model" in i]
        dic_list=sorted(dic_list,key=lambda k:int(k.split(".")[-1]))
        return dir_path+"/"+dic_list[-1]
    def train(self,epoch):
        self.bert_model.train()
        self.iteration(epoch,self.train_dataloader,train=True)
    def iteration(self,epoch,data_loader,train):
        str_code ="train" if train else "test"
        data_iter =tqdm.tqdm(enumerate(data_loader),
                             desc="EP_%s:%d"%(str_code,epoch),
                             total=len(data_loader),
                             bar_format="{l_bar}{r_bar}")

        total_next_sen_loss=0
        total_mlm_loss=0
        total_next_sen_acc=0
        total_mlm_acc=0
        total_element=0
        #[1,max_seq_len,hidden_size]
        #将max_swq_len给替换成betch的最大的长度
        for i,data in data_iter:
            data=self.padding(data)
            data={k:v.to(self.device) for k,v in data.items()}
            positional_enc=self.positional_enc[:,:data["bert_input"].size()[-1],:].to(self.device)
            mlm_preds,next_sen_preds=self.bert_model.forward(input_ids=data["bert_input"],
                                                            positional_enc=positional_enc,
                                                            token_type_ids=data["segment_label"])
            #计算pred的准确率
            mlm_acc=self.get_mlm_accuracy(mlm_preds,data["bert_label"])
            #计算cls的准确率
            next_sen_acc=next_sen_preds.argmax(dim=-1,keepdim=False).eq(data["is_next"]).sum().item()
            #计算损失
            mlm_loss=self.compute_loss(mlm_preds,data["bert_label"],self.vocab_size,ignore_index=0)
            next_sen_loss=self.compute_loss(next_sen_preds,data["is_next"])
            loss=mlm_loss+next_sen_loss
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_next_sen_loss+=next_sen_loss.item()
            total_mlm_loss+=mlm_loss.item()
            total_next_sen_acc+=next_sen_acc
            total_element+=data["is_next"].nelement()
            total_mlm_acc+=mlm_acc
            log_dic = {
                "epoch": epoch,
                "test_next_sen_loss": total_next_sen_loss / (i + 1),
                "test_mlm_loss": total_mlm_loss / (i + 1),
                "test_next_sen_acc": total_next_sen_acc / total_element,
                "test_mlm_acc": total_mlm_acc / (i + 1),
                "train_next_sen_loss": 0, "train_mlm_loss": 0,
                "train_next_sen_acc": 0, "train_mlm_acc": 0
            }
            if i % 10 == 0:
                data_iter.write(str({k: v for k, v in log_dic.items() if v != 0 and k != "epoch"}))

    def compute_loss(self,pred,label,num_class=2,ignore_index=None):
        if ignore_index is None:
            loss_func=CrossEntropyLoss()
        else:
            loss_func=CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(pred.view(-1,num_class),label.view(-1))

    def get_mlm_accuracy(self,pred,labels):
        predictions=torch.argmax(pred,dim=-1,keepdim=False)
        mask=(labels>0).to(self.device)
        #把labels中为0的替换掉，就是不计算
        mlm_accuracy=torch.sum((predictions==labels)*mask).float()
        mlm_accuracy=mlm_accuracy/(torch.sum(mask).float()+1e-8)
        return mlm_accuracy

    def padding(self,data):
        #之所以遍历是可能存在多个样本
        bert_input=[i["bert_input"] for i in data]
        bert_label=[i["bert_label"] for i in data]
        segment_label=[i["segment_label"] for i in data]
        #将其填充成相同的长度，用0来填充
        bert_input=torch.nn.utils.rnn.pad_sequence(bert_input,batch_first=True)
        bert_label=torch.nn.utils.rnn.pad_sequence(bert_label,batch_first=True)
        segment_label=torch.nn.utils.rnn.pad_sequence(segment_label,batch_first=True)
        is_next=torch.cat([i["is_next"] for i in data])
        return {"bert_input": bert_input,
                "bert_label": bert_label,
                "segment_label": segment_label,
                "is_next": is_next}
    def save_state_dict(self,model,epoch,dir_path,file_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        model.to("cpu")
        save_path=dir_path+"/"+file_path+".epoch.{}".format(str(epoch))
        print("{} is saved !".format(save_path))
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        model.to(self.device)
    def test(self,epoch):
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch,self.test_dataloader,train=False)

if __name__ =="__main__":
    def init_trainer(load_model=False):
        trainer=Pertrainer(BertForPreTraining,vocab_size=config["vocab_size"],
                   max_seq_len=config["max_seq_len"],lr=config["lr"],
                   batch_size=config["batch_size"])
        if load_model:
            trainer.load_model(trainer.bert_model,dir_path=config["output_path"])
        return trainer
    trainer=init_trainer(load_model=False)
    start_epoch=1
    train_epoch=200
    for epoch in range(start_epoch,start_epoch+train_epoch):
        trainer.train(epoch)
        trainer.save_state_dict(trainer.bert_model,epoch,dir_path=config["output_path"],
        file_path="bert_model")
        trainer.test(epoch)



