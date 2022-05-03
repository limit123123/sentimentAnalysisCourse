import torch

from RNN import config
from Data import dataLoader
from RNN import RNN
from train_test_RNN import test
import os

if __name__ == '__main__':
    configRNN = config()
    test_data = dataLoader.build_dataset_test(configRNN,configRNN.test_data)
    testDataLoader = dataLoader.DataLoaderTest(test_data,configRNN)
    modelRNN = RNN(configRNN).to(configRNN.device)
    modelRNN.load_state_dict(torch.load(configRNN.save_model_path))
    test(configRNN,modelRNN,testDataLoader)

