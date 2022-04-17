
import torch

class config():
    def __init__(self):
        # 路径相关设置
        self.data_path = "../data/"
        self.train_data = self.data_path + "train_data.json"
        self.eval_data = self.data_path + "eval_data.json"
        self.test_data = self.data_path + ""
        self.model_name = "TextCNN"

        # 分类类别设置
        self.class_data = [x.strip() for x in open(self.data_path + 'class.txt', encoding='utf-8').readlines()]
        self.class_len = len(self.class_data)      # 分类类别数

        # CNN网络结构设置
        self.filter_size = (2, 3, 4)               # 卷积核size的设置
        self.filter_num = 256                      # 卷积核输出的channel数

        # 训练设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.dropout = 0.3                         # 随机丢弃
        self.maxiter_without_improvement = 1000    # 若超过1000轮效果仍然没有提升，则提前结束训练
        self.epoch = 200                           # 训练轮数
        self.learning_rate =1e-3                   # 学习率
        self.save_model_path = "./train/cnn.ckpt"
        self.log_path = "./train/logs"

        # dataloader部分
        self.batch_size = 128                      # 每次取出的数据的量
        self.padding_size = 32                     # 每句话处理成的长度,长切短补

        # embedding词嵌入部分










