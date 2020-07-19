from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
def tokenizer_space(line):
    # 按空格分词
    return [li for li in line.split() if li.strip() != '']
    
    
#训练样本的tf-idf的生成
def get_data_tf_idf(train_text):
    #tokenizer=tokenizer_space表示使用上面的tikenizer_space完成分词任务
    vectoring = TfidfVectorizer(input='content', tokenizer=tokenizer_space, analyzer='word')
    content = open(train_text, 'r', encoding='utf8').readlines()
    x = vectoring.fit_transform(content)
    return x, vectoring

#测试样本的tf-idf的生成    
def get_test_data_tf_idf(train_text,test_text):
    vectoring = TfidfVectorizer(input='content', tokenizer=tokenizer_space, analyzer='word')
    content = open(train_text, 'r', encoding='utf8').readlines()
    content1 = open(test_text, 'r', encoding='utf8').readlines()
    x_train = vectoring.fit_transform(content)
    x_test = vectoring.transform(content1)
    return x_test, vectoring

#获取样本的标签
def get_label_list(label_file_name):
     with open(label_file_name, 'r', encoding='utf8') as f:
         lebel_list=[]
         for line in f:
             lebel_list.append(line[0])
     return np.array(lebel_list)
train_text = 'email.txt'
train_labels = 'labels.txt'


x, vectoring = get_data_tf_idf(train_text)

y = get_label_list(train_labels)


#样本打乱
index = np.arange(len(y))  
np.random.shuffle(index)
x = x[index]
y = y[index]

#8:2划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
learning_rate=[0.001,0.001,0.01,0.1,0.3,0.9]
model=XGBClassifier() 
param_grid=dict(learning_rate=learning_rate)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=7)
grid_search=GridSearchCV(model,param_grid,scoring='neg_log_loss',n_jobs=-1,cv=kfold)
    
grid_result=grid_search.fit(x_train, y_train)      
print("Best:%f using %s"%(grid_result.best_score_,grid_result.best_params_))
