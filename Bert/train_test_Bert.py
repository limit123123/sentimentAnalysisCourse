# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from optimization import BertAdam
from torch.optim import AdamW

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config, model,train_iter,dev_iter):
    """
    模型训练开始
    """
    start_time = time.time()
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    # 记录进行到多少个batch
    total_batch = 0
    dev_best_loss = float('inf')
    dev_best_acc = float(0)
    # 记录上次验证集loss下降
    last_improve = 0
    # 记录当前效果是否提升
    flag = False
    model.train()
    for epoch in range(config.epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, config.epoch))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels.long())
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                ground_truth = labels.data.cpu()
                predict_labels = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(ground_truth, predict_labels)
                dev_acc, dev_loss = evaluate(model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.model_saved_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                print("Iter:{:4d} TrainLoss:{:.10f} TrainAcc:{:.5f} DevLoss:{:.12f} DevAcc:{:.5f} Improve:{}".format(total_batch,loss.item(),train_acc,dev_loss,dev_acc,improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.maxiter_without_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    end_time = time.time()
    print("Train Time : {:.3f} min , The Best Acc in Dev : {} %".format(((float)((end_time-start_time))/60), dev_best_acc))


def test(config,model, test_iter):
    """
    模型测试
    """
    start_time = time.time()
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, (trains) in enumerate(test_iter):
            outputs = model(trains)
            predict_labels = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predict_labels)
    end_time = time.time()
    print("Predict Time : {} s".format(((float)(end_time - start_time))))
    predict_result = np.column_stack([test_iter.ids, predict_all])
    np.savetxt(config.predict_save, predict_result, delimiter=',')



def evaluate(model, data_iter):
    """
    使用验证集评估
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels.long())
            loss_total += loss
            ground_truth = labels.data.cpu().numpy()
            predict_labels = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, ground_truth)
            predict_all = np.append(predict_all, predict_labels)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(data_iter)


