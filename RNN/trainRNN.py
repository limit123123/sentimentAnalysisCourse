from RNN import config
from Data import dataLoader
from RNN import RNN
from train_test_RNN import train
import os


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    configRNN = config()
    train_data = dataLoader.build_dataset(configRNN, configRNN.train_data)
    dev_data = dataLoader.build_dataset(configRNN, configRNN.eval_data)
    trainDataLoader = dataLoader.DataLoader(train_data,configRNN)
    devDataLoader = dataLoader.DataLoader(dev_data,configRNN)
    modelRNN = RNN(configRNN).to(configRNN.device)
    train(configRNN,modelRNN,trainDataLoader,devDataLoader)
