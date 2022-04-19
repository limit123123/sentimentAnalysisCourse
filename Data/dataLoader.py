import json
import numpy as np
import torch
from tqdm import tqdm
import jieba

PAD = '<PAD>'
UNK = '<UNK>'


# 情感分析与计算的数据集及dataloader
def build_dataset(config,data_path):
    """
    构建训练集和验证集的dataLoader
    """
    labelMap = {'positive':1,'negative':2,'neutral':0}
    contents = []
    if config.token == 'word':
        for word in config.vocab:
            jieba.add_word(word)
    with open(data_path, 'r', encoding='UTF-8') as f:
        dataAll = json.load(f)
        for data in tqdm(dataAll):
            content = data['content'].strip()
            label = labelMap[data['label']]
            if config.token == 'character':
                # 基于字切分
                token = [word for word in content]
            else:
                # 基于词切分
                token = list(jieba.cut(content,cut_all=False))
            seq_len = len(token)
            if seq_len < config.padding_size:
                token.extend([PAD] * (config.padding_size - len(token)))
            else:
                token = token[:config.padding_size]
                seq_len = config.padding_size
            word_ids = [config.vocab_dict.get(word,config.vocab_dict.get(UNK)) for word in token]
            contents.append((word_ids, int(label), seq_len))
    return contents

def build_dataset_test(config,data_path):
    """
    构建测试集的dataset，和训练集、验证集的dataset主要区别为不含labels
    """
    contents = []
    ids = np.array([], dtype=int)
    with open(data_path, 'r', encoding='UTF-8') as f:
        dataAll = json.load(f)
        for data in tqdm(dataAll):
            id = data['id']
            content = data['content'].strip()
            # 获取所有字
            token = [word for word in content]
            seq_len = len(token)
            if seq_len < config.padding_size:
                token.extend([PAD] * (config.padding_size - len(token)))
            else:
                token = token[:config.padding_size]
                seq_len = config.padding_size
            ids = np.append(ids,id)
            word_ids = [config.vocab_dict.get(word,config.vocab_dict.get(UNK)) for word in token]
            contents.append((word_ids, seq_len))
    return (contents,ids)



class DataLoader(object):
    def __init__(self, dataset,config):
        """
        :param dataset:dataset所有数据集
        :param config: 相关配置
        """
        # batches为数据集
        self.batches = dataset
        # batch_size为打包的数据大小
        self.batch_size = config.batch_size
        # n_batch为打包的个数
        self.n_batches = len(self.batches) // self.batch_size
        self.residue = False
        if len(self.batches) != self.batch_size*self.n_batches:
            self.residue = True
        self.index = 0
        self.device = config.device

    def ToTensor(self,dataset):
        """
        :param dataset:batches截取后的段
        :return: tensor
        """
        X = torch.Tensor([_[0] for _ in dataset]).long().to(self.device)
        y = torch.Tensor([_[1] for _ in dataset]).long().to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in dataset]).long().to(self.device)

        return (X, seq_len), y

    def __next__(self):
        """
        DataLoader迭代器
        """
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self.ToTensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self.ToTensor(batches)
            return batches

    def __iter__(self):
        """
        :return:返回本身
        """
        return self

    def __len__(self):
        """
        :return: 返回长度
        """
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


class DataLoaderTest(object):
    def __init__(self, dataset,config):
        """
        :param dataset: dataset所有数据集
        :param config: 相关配置
        """
        # batches为数据集
        self.batches = dataset[0]
        self.ids = dataset[1]
        # batch_size为打包的数据大小
        self.batch_size = config.batch_size
        # n_batch为打包的个数
        self.n_batches = len(self.batches) // self.batch_size
        self.residue = False
        if len(self.batches) != self.batch_size*self.n_batches:
            self.residue = True
        self.index = 0
        self.device = config.device

    def ToTensor(self,dataset):
        """
        :param dataset:batches截取后的段
        :return: tensor
        """
        X = torch.Tensor([_[0] for _ in dataset]).long().to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in dataset]).long().to(self.device)
        return (X, seq_len)

    def __next__(self):
        """
        DataLoader迭代器
        """
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self.ToTensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self.ToTensor(batches)
            return batches

    def __iter__(self):
        """
        :return:返回本身
        """
        return self

    def __len__(self):
        """
        :return: 返回长度
        """
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches



