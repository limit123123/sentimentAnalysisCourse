import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import word2vec


''' RNN的相关配置文件 '''
class config(object):
    def __init__(self):
        # 路径相关设置
        self.data_path = "../Data/Dataset/"
        self.embedding_path = "../Data/Embedding/"
        self.train_data = self.data_path + "train_data.json"
        self.eval_data = self.data_path + "eval_data.json"
        self.test_data = self.data_path + "test_data.json"
        self.model_name = "TextRNN"
        self.predict_save = "./test/predict.csv"

        # 分类类别设置
        self.class_data = [x.strip() for x in open(self.data_path + 'class.txt', encoding='utf-8').readlines()]
        self.class_len = len(self.class_data)      # 分类类别数

        # RNN网络结构设置
        self.hidden_size = 32                     # 隐藏层
        self.layer_nums = 2                        # rnn层数
        self.dropout = 0.3  # 随机丢弃

        # 训练设置
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')  # 设备
        # self.device = torch.device('cpu')  # 设备
        self.maxiter_without_improvement = 1000    # 若超过1000轮效果仍然没有提升，则提前结束训练
        self.epoch = 200                           # 训练轮数
        self.learning_rate =1e-3                   # 学习率


        # dataloader部分
        self.batch_size = 128                      # 每次取出的数据的量
        self.padding_size = 32                     # 每句话处理成的长度,长切短补

        # embedding词嵌入部分
        self.embedding = 'NotRandom'
        if self.embedding == 'NotRandom':
            self.token = 'word'  # 基于词级别做切分
            embedding_model = word2vec.load(self.embedding_path + "sgns.weibo.word.txt")  # 词嵌入模型
            self.vocab = embedding_model.vocab  # 词表
            self.vocab_dict = embedding_model.vocab_hash  # 词表对应的编号
            self.vocab_dict.update({'<UNK>': len(self.vocab), '<PAD>': len(self.vocab) + 1})  # 补充未登录词及PAD
            self.vocab = np.append(np.append(self.vocab, '<UNK>'), '<PAD>')  # 补充未登录词
            self.vectors = embedding_model.vectors  # vocab_size * embedding_size大小的矩阵
            self.vectors = torch.Tensor(np.append(np.append(self.vectors, self.vectors.mean(axis=0).reshape(1, -1),
                                                            axis=0), self.vectors.mean(axis=0).reshape(1, -1), axis=0))
            self.embedding_size = self.vectors.shape[1]  # 字处理成的向量的维度
            self.vocab_size = self.vectors.shape[0]  # 词表的大小
        else:
            self.token = 'character'  # 基于字的级别做切分
            self.embedding_size = 300  # 字向量embedding的大小
            self.vocab_dict = {}  # 词对应的编号
            with open(self.train_data, 'r', encoding='UTF-8') as f:
                for jsonData in json.load(f):
                    for word in jsonData['content']:
                        self.vocab_dict[word] = self.vocab_dict.get(word, 0) + 1
            self.vocab = sorted([_ for _ in self.vocab_dict.items()], key=lambda x: x[1], reverse=True)
            self.vocab_dict = {word[0]: idx for idx, word in enumerate(self.vocab)}
            self.vocab_dict['<UNK>'] = len(self.vocab_dict)
            self.vocab_dict['<PAD>'] = len(self.vocab_dict)
            self.vocab = list(self.vocab_dict.keys())
            self.vocab_size = len(self.vocab)  # 词表的大小，运行时设置
        # 模型保存路径
        self.save_model_path = "./train/models/" + self.embedding + "RNN.ckpt"
        self.log_path = "./train/logs/"+ self.embedding + '/'


class RNN(nn.Module):
    def __init__(self,config):
        super(RNN,self).__init__()
        if config.embedding != 'Random':
            self.Embedding = nn.Embedding.from_pretrained(config.vectors,freeze=False)
        else:
            self.Embedding = nn.Embedding(config.vocab_size,config.embedding_size)
        self.LSTM = nn.LSTM(config.embedding_size, config.hidden_size, num_layers=config.layer_nums, bidirectional=True,batch_first=True,dropout=config.dropout)
        self.Fc = nn.Linear(2*config.hidden_size, config.class_len)

    def forward(self, x):
        x, _ = x                             # _为seq_len
        output = self.Embedding(x)           # output.shape [batch_size, seq_len, Embeding] = [batch_size, 32, 300]
        output, _ = self.LSTM(output)        # output.shape [batch_size, seq_len, 2*hidden_size] = [batch_size, 32, 64]
        output = self.Fc(output[:, -1, :])  # 取最后时刻的隐藏状态 hidden state
        return output



























