import configparser
import json
from models.bert_sentiment_analysis import *
import torch
import numpy as np
import os
import tqdm
from sklearn import metrics
from metrics import *
from torch.utils.data import DataLoader
from dataset.sentiment_dataset import  CLSDataset
class Sentiment_trainer:
    def __init__(self,max_seq_len,
                 batch_size,lr,use_cuda=False):
        self.max_seq_len=max_seq_len
        self.batch_size=batch_size
        self.lr=lr
        self.use_cuda=use_cuda
        config_=configparser.ConfigParser()
        config_.read("./config/sentiment_model_config.ini")
        self.config=config_["DEFAULT"]
        with open(self.config["word2idx_path"],"r",encoding="utf-8") as f:
            self.word2idx=json.load(f)
        bertConfig =BertConfig(vocab_size=int(self.config["vocab_size"]))
        #print(self.config["vocab_size"])
        self.bert_model=Bert_Sentiment_Analysis(config=bertConfig)
        #print("aaa")
        use_cuda=torch.cuda.is_available() and use_cuda
        
        self.device=torch.device("cuda:0" if use_cuda else "cpu")
        self.bert_model.to(self.device)

        train_dataset=CLSDataset(corpus_path=self.config["train_corpus_path"],word2idx=self.word2idx,
                                 max_seq_len=self.max_seq_len,data_regularization=True)
        self.train_dataloader=DataLoader(train_dataset,batch_size=self.batch_size,num_workers=0,collate_fn=lambda x:x)

        test_dataset=CLSDataset(corpus_path=self.config["test_corpus_path"],word2idx=self.word2idx,
                                max_seq_len=self.max_seq_len,data_regularization=False)
        self.test_dataloader=DataLoader(test_dataset,batch_size=self.batch_size,num_workers=0,collate_fn=lambda x:x)

        self.hidden_size=bertConfig.hidden_size
        #位置编码
        self.positional_enc=self.init_positional_encoding()
        self.positional_enc=torch.unsqueeze(self.positional_enc,dim=0)
        self.optim_parameters=list(self.bert_model.parameters())
        self.init_optimizer(lr=self.lr)
        if not os.path.exists(self.config["state_dict_dir"]):
            os.mkdir(self.config["state_dict_dir"])
    def init_optimizer(self,lr):
        self.optimizer=torch.optim.Adam(self.optim_parameters,lr=lr,weight_decay=1e-3)
    def init_positional_encoding(self):
        pos_enc=np.array([
            [pos/np.power(10000,2*i/self.hidden_size) for i in range(self.hidden_size)]
            if pos!=0 else np.zeros(self.hidden_size) for pos in range(self.max_seq_len)
        ])
        pos_enc[1:,0::2]=np.sin(pos_enc[1:,0::2])
        pos_enc[1:,1::2]=np.cos(pos_enc[1:,1::2])
        denomiator=np.sqrt(np.sum(pos_enc**2,axis=1,keepdims=True))
        #归一化
        position_enc=pos_enc/(denomiator+1e-8)
        position_enc=torch.from_numpy(position_enc).type(torch.FloatTensor)
        #print(position_enc.shape)[hidden_dim,max_seq_len]
        return position_enc





    def padding(self,data):
        text_input=[i["text_input"] for i in data]
        text_input=torch.nn.utils.rnn.pad_sequence(text_input,batch_first=True)
        label=torch.cat([i["label"] for i in data])
        return {
            "text_input": text_input,
            "label":label
        }

    def iteration(self,epcoh,data_loader,train):
        str_code="train" if train else "test"
        data_iter=tqdm.tqdm(enumerate(data_loader),desc="EP_%s:%d"%(str_code,epoch),
                            total=len(data_loader),bar_format="{l_bar}{r_bar}")
        total_loss=0
        all_predictions,all_labels=[],[]
        for i,data in data_iter:
            data=self.padding(data)
            data={k: v.to(self.device) for k,v in data.items()}
            positional_enc=self.positional_enc[:,:data["text_input"].size()[-1],:].to(self.device)
            predictions,loss=self.bert_model.forward(text_input=data["text_input"],positional_enc=positional_enc,labels=data["label"])
            predictions=predictions.detach().cpu().numpy().reshape(-1).tolist()
            labels=data["label"].cpu().numpy().reshape(-1).tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            fpr,tpr,threshold=metrics.roc_curve(y_true=all_labels,y_score=all_predictions)
            auc=metrics.auc(fpr,tpr)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss+loss.item()
            threshold_=find_best_threshold(all_predictions,all_labels)
            print(str_code+"best threshold" +str(threshold))
        if not train:
            return auc
    def train(self,epcoch):
        self.bert_model.train()
        self.iteration(epoch,self.train_dataloader,train=True)
    def test(self,epoch):
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch,self.test_dataloader,train=False)

    def find_most_recent_state_dict(self,dir_path):
        dic_list=[i for i in os.listdir(dir_path)]
        if len(dic_list)==0:
            raise FileNotFoundError("cat not find file")
        dic_list=[i for i in dic_list if "model" in i]
       
        dic_list=sorted(dic_list,key=lambda k: int(k.split(".")[-1]))
        return dir_path+"/"+dic_list[-1]
    #加载模型
    def load_model(self,bert_model,dir_path,load_model=False):
        checkpoint_dir=self.find_most_recent_state_dict(dir_path)
        checkpoint=torch.load(checkpoint_dir)
        if load_model:
            checkpoint["model_state_dict"]={k[5:]:v for k,v in checkpoint["model_state_dict"].items() if k[:4]=="bert" and "pooler" not in k}
        bert_model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        torch.cuda.empty_cache()
        bert_model.to(self.device)
    #保存模型
    def save_state_dict(self,bert_model,epcoh,state_dict_dir,file_path):
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)
        save_path=state_dict_dir+"/"+file_path+".epoch.{}".format(str(epoch))
        bert_model.to("cpu")
        torch.save({"model_state_dict":bert_model.state_dict()},save_path)



if __name__=="__main__":
    def init_trainer(dynamic_lr,batch_size):
        trainer=Sentiment_trainer(max_seq_len=300,batch_size=batch_size,
                                  lr=dynamic_lr,use_cuda=False)
        return trainer,dynamic_lr
    start_epoch=0
    train_epoches=10000000
    trainer,dynamic_lr=init_trainer(dynamic_lr=1e-06,batch_size=2)
    all_auc=[]
    threshold =999
    patient=10
    best_loss=9999999
    for epoch in range(start_epoch,start_epoch+train_epoches):
        #第一次epoch，并且epoch=0.此时我们需要加载模型
        if epoch==start_epoch and epoch==0:
            trainer.load_model(trainer.bert_model,dir_path="./output_wiki_bert",load_model=True)
        elif epoch ==start_epoch:
            trainer.load_model(trainer.bert_model,dir_path=trainer.config["state_dict_dir"])
        trainer.train(epoch)
        trainer.save_state_dict(trainer.bert_model,epoch,state_dict_dir=trainer.config["state_dict_dir"],
                              file_path="sentiment.model")
        
        auc=trainer.test(epoch)
        all_auc.append(auc)
        best_auc=max(all_auc)
        if all_auc[-1]<best_auc:
            threshold+=1
            dynamic_lr*=0.8
            trainer.init_optimizer(lr=dynamic_lr)
        else:
            threshold=0
            #只要有一次auc变好了，那么就可以置为0了
        if threshold >= patient:
            print("epoch {} has the lowest loss".format(start_epoch + np.argmax(np.array(all_auc))))
            print("early stop!")
            break


