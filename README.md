# 本项目为哈尔滨工业大学情感分析与计算实验
本系统主要采用了文本分类的方法，项目由传统神经网络如CNN、RNN等和基于Bert预训练模型的方法构成。  

# 数据集
采用学校老师发放的微博数据集，为json格式，训练集的大小为8000，验证集的大小为2000，测试集大小为3000，格式如下:  
```json
[
  {
    "id": 1,
    "content": "天使",
    "label": "positive"
  },
  {
    "id": 2,
    "content": "致敬[心][心]小凡也要做好防护措施哦//@Mr_凡先生:致敬[心]大家出门记得戴口罩",
    "label": "positive"
  },
  {
    "id": 3,
    "content": "[中国赞][中国赞][中国赞]",
    "label": "positive"
  }
]
```
需要数据集请私信xiong257246@outlook.com。  

# 数据预处理
使用基于




# 实验结果 
TODO

|模型|Accuracy|
|:---:|:---:|
|CNN|%|
|Bert|%|


# 实验
## CNN文本分类
### CNN文本分类内容
对应参考文献[1]  
### CNN文本分类实验结果
|模型|Accuracy|
|:---:|:---:|
|CNN|71.45%|
|SGNS+CNN|72.85%|


##


## 基于Bert预训练模型





# 致谢
部分内容参考  
[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)  
[Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)  
[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
# 参考文献
[1] Convolutional Neural Networks for Sentence Classification  
[2] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  




