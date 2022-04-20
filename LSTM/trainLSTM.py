from LSTM import config
from Data import dataLoader
from LSTM import LSTM
from train_test_LSTM import train
import os


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    configLSTM = config()
    train_data = dataLoader.build_dataset(configLSTM, configLSTM.train_data)
    dev_data = dataLoader.build_dataset(configLSTM, configLSTM.eval_data)
    trainDataLoader = dataLoader.DataLoader(train_data,configLSTM)
    devDataLoader = dataLoader.DataLoader(dev_data,configLSTM)
    modelLSTM = LSTM(configLSTM).to(configLSTM.device)
    train(configLSTM,modelLSTM,trainDataLoader,devDataLoader)
