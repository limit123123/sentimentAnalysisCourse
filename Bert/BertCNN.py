from transformers import BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoConfig,AutoModel,BertConfig

class config(object):
    def __init__(self):
        # 定义相关路径
        self.tokenizer_path = '../BertPretrained'                   # Bert的tokenizer路径
        self.model_path = '../BertPretrained'                       # Bert模型路径
        self.model_name = 'BertCNN'                        # Bert模型的名称
        self.data_path = '../Data/Dataset/'                          # 数据的顶层路径
        self.train_data = self.data_path + 'train_data.json'         # 训练集
        self.eval_data = self.data_path + 'eval_data.json'           # 验证集
        self.test_data = self.data_path + 'test_data.json'           # 测试集
        self.predict_save = "./test/predictBertCNN.csv"                     # 定义测试集结果保存路径
        self.model_saved_path = './models/'+self.model_name+'.ckpt'  # 定义模型保存的路径及名称
        self.model_config_path = self.model_path + '/config.json'     # 定义模型的相关配置文件路径

        # 分类类别设置
        self.class_data = [x.strip() for x in open(self.data_path + 'class.txt', encoding='utf-8').readlines()]
        self.class_len = len(self.class_data)                        # 分类类别数

        # 训练设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.maxiter_without_improvement = 1000                      # 若超过1000轮效果仍然没有提升，则提前结束训练
        self.epoch = 200                                             # 训练轮数
        self.learning_rate = 1e-3                                    # 学习率
        self.batch_size = 64                                         # mini-batch的大小，即一次训练的大小
        self.pad_size = 64                                           # 每句话处理成的长度
        self.learning_rate = 5e-5                                    # 学习率
        self.hidden_size = 768                                       # 隐藏层大小

        # 读入bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)

        # CNN网络结构设置
        self.filter_size = (2, 3, 4)                 # 卷积核size的设置
        self.filter_num = 256                        # 卷积核输出的channel数
        self.dropout = 0.3


class BertCNN(nn.Module):

    def __init__(self, config):
        super(BertCNN, self).__init__()
        bertConfig = BertConfig.from_json_file(config.model_config_path)
        # 读入bert model
        self.bert = BertModel.from_pretrained(config.model_path, config=bertConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.Convs = nn.ModuleList([nn.Conv2d(1, config.filter_num, (k, config.hidden_size)) for k in config.filter_size])
        self.Dropout = nn.Dropout(config.dropout)
        self.Fc = nn.Linear(config.filter_num * len(config.filter_size), config.class_len)


    def forward(self, x):
        context = x[0]
        mask = x[2]
        output = self.bert(context.long(), attention_mask=mask)
        output = output['last_hidden_state']                 # [batch_size, seq_len, hidden_size]
        output = output.unsqueeze(1)
        outputList = []
        for Conv in self.Convs:
            outputConv = Conv(output).squeeze(3)
            outputMaxpool = F.max_pool1d(outputConv,outputConv.size(2)).squeeze(2)
            outputList.append(outputMaxpool)
        output = torch.cat(outputList,dim=1)
        output = self.Dropout(output)
        output = self.Fc(output)
        return output