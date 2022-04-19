import torch

from Bert import config
import Data.dataLoaderBert as dataLoader
from Bert import Bert
from train_test_Bert import test
import os

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    configBert = config()
    test_data = dataLoader.build_dataset_test(configBert,configBert.test_data)
    testDataLoader = dataLoader.DataLoaderTest(test_data,configBert)
    modelBert = Bert(configBert).to(configBert.device)
    modelBert.load_state_dict(torch.load(configBert.model_saved_path))
    test(configBert,modelBert,testDataLoader)

