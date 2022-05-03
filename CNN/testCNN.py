import torch

from CNN import config
from Data import dataLoader
from CNN import CNN
from train_test_CNN import test
import os

if __name__ == '__main__':
    configCNN = config()
    test_data = dataLoader.build_dataset_test(configCNN,configCNN.test_data)
    testDataLoader = dataLoader.DataLoaderTest(test_data,configCNN)
    modelCNN = CNN(configCNN).to(configCNN.device)
    modelCNN.load_state_dict(torch.load(configCNN.save_model_path))
    test(configCNN,modelCNN,testDataLoader)

