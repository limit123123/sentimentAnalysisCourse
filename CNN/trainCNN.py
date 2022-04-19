from CNN import config
from Data import dataLoader
from CNN import CNN
from train_test_CNN import train
import os

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    configCNN = config()
    train_data = dataLoader.build_dataset(configCNN, configCNN.train_data)
    dev_data = dataLoader.build_dataset(configCNN, configCNN.eval_data)
    trainDataLoader = dataLoader.DataLoader(train_data,configCNN)
    devDataLoader = dataLoader.DataLoader(dev_data,configCNN)
    modelCNN = CNN(configCNN).to(configCNN.device)
    train(configCNN,modelCNN,trainDataLoader,devDataLoader)
