from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import json
from sklearn.metrics import f1_score
import os

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
        self.model_path = './pretrainedModel'
        self.tokenizer_path = './pretrainedModel'

        # 训练相关设置
        self.learning_rate = 1e-6   # 学习率
        self.epoch = 20             # 训练轮数
        self.batch_size = 32        # batch_size
        self.output_path = './outputs'




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
        else:
            with open(path, 'r', encoding='UTF-8') as f:
                dataAll = json.load(f)
                self.contents = [(data['content'].strip()) for data in dataAll]
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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    micro_f1_score = f1_score(labels, predictions, average='micro')
    macro_f1_score = f1_score(labels, predictions, average='macro')
    return {
        'Micro f1 score': micro_f1_score,
        'Macro f1 score': macro_f1_score,
        'eval_f1_score': macro_f1_score,
    }


if __name__ == '__main__':
    configModel = config()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = AutoModelForSequenceClassification.from_pretrained(configModel.model_path, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(configModel.model_path)
    trainDataset = myDataset(configModel.train_path, tokenizer)
    evalDataset = myDataset(configModel.eval_path, tokenizer)
    train_args = TrainingArguments(
        output_dir=configModel.output_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=configModel.learning_rate,
        per_device_train_batch_size=configModel.batch_size,
        per_device_eval_batch_size=configModel.batch_size,
        num_train_epochs=configModel.epoch,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        eval_accumulation_steps=3,
        save_total_limit=1
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=trainDataset,
        eval_dataset=evalDataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.evaluate()
    trainer.train()
