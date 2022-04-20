import torch
import Data.dataLoaderBert as dataLoader
from train_test_Bert import test
import Bert
import BertCNN
import os
import argparse

parser = argparse.ArgumentParser(description='Bert Text Classification')
parser.add_argument('--model', type=str, required=True, help='make a choice: Bert、BertCNN')
args = parser.parse_args()

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    modelName = args.model
    if modelName == 'BertCNN':
        configBert = BertCNN.config()
        BertModel = BertCNN.BertCNN(configBert).to(configBert.device)
    else:
        configBert = Bert.config()
        BertModel = Bert.Bert(configBert).to(configBert.device)


    test_data = dataLoader.build_dataset_test(configBert,configBert.test_data)
    testDataLoader = dataLoader.DataLoaderTest(test_data,configBert)
    BertModel.load_state_dict(torch.load(configBert.model_saved_path))
    test(configBert, BertModel,testDataLoader)

