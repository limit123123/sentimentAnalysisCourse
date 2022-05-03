import torch

from LSTM import config
from Data import dataLoader
from LSTM import LSTM
from train_test_LSTM import test
import os

if __name__ == '__main__':
    configLSTM = config()
    test_data = dataLoader.build_dataset_test(configLSTM,configLSTM.test_data)
    testDataLoader = dataLoader.DataLoaderTest(test_data,configLSTM)
    modelLSTM = LSTM(configLSTM).to(configLSTM.device)
    modelLSTM.load_state_dict(torch.load(configLSTM.save_model_path))
    test(configLSTM,modelLSTM,testDataLoader)

