import json
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import numpy as np
PAD = '[PAD]'
CLS = '[CLS]'

def build_dataset(config, path):
    labelMap = {'positive':1, 'negative':2, 'neutral':0}
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        dataAll = json.load(f)
        for data in tqdm(dataAll):
            content = data['content'].strip()
            label = labelMap[data['label']]
            token = config.tokenizer.tokenize(content)
            # 由于pad为0，因此最后不用补
            token = [CLS] + token
            seq_len = len(token)
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            if len(token) < config.pad_size:
                mask = [1] * len(token_ids) + [0] * (config.pad_size - len(token))
                token_ids += ([0] * (config.pad_size - len(token)))
            else:
                mask = [1] * config.pad_size
                token_ids = token_ids[:config.pad_size]
                seq_len = config.pad_size
            contents.append((token_ids, int(label), seq_len, mask))
    return contents

def build_dataset_test(config, path):
    contents = []
    ids = np.array([], dtype=int)
    with open(path, 'r', encoding='UTF-8') as f:
        dataAll = json.load(f)
        for data in tqdm(dataAll):
            content = data['content'].strip()
            token = config.tokenizer.tokenize(content)
            # 由于pad为0，因此最后不用补
            token = [CLS] + token
            seq_len = len(token)
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            if len(token) < config.pad_size:
                mask = [1] * len(token_ids) + [0] * (config.pad_size - len(token))
                token_ids += ([0] * (config.pad_size - len(token)))
            else:
                mask = [1] * config.pad_size
                token_ids = token_ids[:config.pad_size]
                seq_len = config.pad_size
            ids = np.append(ids,data['id'])
            contents.append((token_ids, seq_len, mask))
    return (contents,ids)




class DataLoader(object):
    def __init__(self, dataset, config):
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
        X = torch.Tensor([_[0] for _ in dataset]).to(self.device)
        y = torch.Tensor([_[1] for _ in dataset]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in dataset]).to(self.device)
        # 1的部分为有值的部分
        mask = torch.LongTensor([_[3] for _ in dataset]).to(self.device)
        return (X, seq_len, mask), y

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
    def __init__(self, dataset, config):
        """
        :param dataset:dataset所有数据集
        :param Config: 相关配置
        """
        # batches为数据集
        self.batches = dataset[0]
        # ids
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
        X = torch.Tensor([_[0] for _ in dataset]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in dataset]).to(self.device)
        # 1的部分为有值的部分
        mask = torch.LongTensor([_[2] for _ in dataset]).to(self.device)
        return (X, seq_len, mask)

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

