from torch import nn
from models.bert_model import *

class Bert_Sentiment_Analysis(nn.Module):
    def __init__(self,config):
        super(Bert_Sentiment_Analysis,self).__init__()
        self.bert=BertModel(config)
        self.dense=nn.Linear(config.hidden_size*2,config.hidden_size)
        self.final_dense=nn.Linear(config.hidden_size,1)
        self.activation=nn.Sigmoid()
    def compute_loss(self,predictions,labels=None):
        predictions=predictions.view(-1)
        labels=labels.float().view(-1)
        epsilon=1e-8
        loss = \
            - labels * torch.log(predictions + epsilon) - \
            (torch.tensor(1.0) - labels) * torch.log(torch.tensor(1.0) - predictions + epsilon)
        # 求均值, 并返回可以反传的loss
        # loss为一个实数
        loss = torch.mean(loss)
        return loss
    def forward(self,text_input,positional_enc,labels=None):
        encoded_layers,_=self.bert(text_input,positional_enc,
                                   output_all_encoded_layers=True)
        sequence_output=encoded_layers[2]
        avg_pooled=sequence_output.mean(1)
        max_pooled=torch.max(sequence_output,dim=1)
        pooled=torch.cat((avg_pooled,max_pooled[0]),dim=1)
        pooled=self.dense(pooled)
        predictions=self.final_dense(pooled)
        predictions=self.activation(predictions)
        if labels is not None:
            loss=self.compute_loss(predictions,labels)
            return predictions,loss

