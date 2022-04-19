# coding: UTF-8
from Bert import config
import time
import os
import numpy as np
import torch
import Data.dataLoaderBert as dataLoader
from Bert import Bert
from train_test_Bert import train
if __name__ == '__main__':

    configBert = config()
    print("Loading data ...")
    train_data = dataLoader.build_dataset(configBert,configBert.train_data)
    dev_data = dataLoader.build_dataset(configBert,configBert.eval_data)
    train_iter = dataLoader.DataLoader(train_data,configBert)
    dev_iter = dataLoader.DataLoader(dev_data,configBert)
    BertModel = Bert(configBert).to(configBert.device)
    train(configBert,BertModel,train_iter,dev_iter)

