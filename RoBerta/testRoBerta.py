from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import json
import os
from sklearn.metrics import f1_score
import pandas as pd

class config():
    """
    模型相关配置
    """
    def __init__(self):
        # 数据相关路径
        self.input_path = '../Data/Dataset/'
        self.train_path = self.input_path + 'train_data.json'
        self.eval_path = self.input_path + 'eval_data.json'
        self.test_path = self.input_path + 'test_data.json'
        # 模型相关数据
        self.model_path = './outputs/checkpoint-5380'
        self.tokenizer_path = './outputs/checkpoint-5380'

        # 训练相关设置
        self.learning_rate = 1e-6   # 学习率
        self.epoch = 50              # 训练轮数
        self.batch_size = 32        # batch_size
        self.output_path = './outputs'
        self.predict_save = './test/predictRoBerta.csv'




class myDataset(Dataset):
    def __init__(self, path, tokenizer, with_label=True):
        super(myDataset, self).__init__()
        self.with_label = with_label
        self.tokenizer = tokenizer
        if self.with_label == True:
            self.labelMap = {'positive': 1, 'negative': 2, 'neutral': 0}
            self.contents = []
            with open(path, 'r', encoding='UTF-8') as f:
                dataAll = json.load(f)
                self.contents = [(data['content'].strip(), self.labelMap[data['label']]) for data in dataAll]
                self.ids = [data['id'] for data in dataAll]
        else:
            with open(path, 'r', encoding='UTF-8') as f:
                dataAll = json.load(f)
                self.contents = [(data['content'].strip()) for data in dataAll]
                self.ids = [data['id'] for data in dataAll]
    def __getitem__(self, item):
        inputs = self.tokenizer(self.contents[item][0], max_length=380, padding='max_length', truncation=True)
        if self.with_label:
            return {
                **inputs,
                'label': self.contents[item][1]
            }
        else:
            return {
                **inputs,
            }
    def __len__(self):
        return len(self.contents)




if __name__ == '__main__':
    configModel = config()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = AutoModelForSequenceClassification.from_pretrained(configModel.model_path)
    tokenizer = AutoTokenizer.from_pretrained(configModel.tokenizer_path)
    testDataset = myDataset(configModel.test_path, tokenizer, with_label=False)
    trainer = Trainer(model, tokenizer=tokenizer)
    outputs = trainer.predict(testDataset)
    outputs = np.argmax(outputs.predictions,axis=1)
    predict_result = np.column_stack((testDataset.ids, outputs))
    np.savetxt(configModel.predict_save, predict_result.astype(int), delimiter=',', fmt='%d,%d', encoding='utf-8')