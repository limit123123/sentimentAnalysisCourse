import json

import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm


PAD = '<PAD>'
UNK = '<UNK>'


# 情感分析与计算的数据集及dataloader
def build_dataset(config,data_path):
    labelMap = {'positive':1,'negative':2,'neutral':0}
    contents = []
    with open(data_path, 'r', encoding='UTF-8') as f:
        dataAll = json.load(f)
        for data in tqdm(dataAll):
            content = data['content'].strip()
            label = labelMap[data['label']]
            # 获取所有字
            token = [word for word in content]
            seq_len = len(token)

            if seq_len < config.padding_size:
                token.extend([PAD] * (config.padding_size - len(token)))
            else:
                token = token[:config.padding_size]
                seq_len = config.padding_size
            word_ids = [config.vocab_dict.get(word,config.vocab_dict.get(UNK)) for word in token]
            contents.append((word_ids, int(label), seq_len))
    return contents

class DataLoader(object):
    def __init__(self, dataset,config):
        """
        :param dataset:dataset所有数据集
        :param Config: 相关配置
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

