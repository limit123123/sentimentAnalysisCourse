# coding: UTF-8
import Data.dataLoaderBert as dataLoader
from train_test_Bert import train
import Bert
import BertCNN
import argparse

parser = argparse.ArgumentParser(description='Bert Text Classification')
parser.add_argument('--model', type=str, required=True, help='make a choice: Bert、BertCNN')
args = parser.parse_args()

if __name__ == '__main__':
    modelName = args.model
    if modelName == 'BertCNN':
        configBert = BertCNN.config()
        BertModel = BertCNN.BertCNN(configBert).to(configBert.device)
    else:
        configBert = Bert.config()
        BertModel = Bert.Bert(configBert).to(configBert.device)
    print("Loading data ...")
    train_data = dataLoader.build_dataset(configBert,configBert.train_data)
    dev_data = dataLoader.build_dataset(configBert,configBert.eval_data)
    train_iter = dataLoader.DataLoader(train_data,configBert)
    dev_iter = dataLoader.DataLoader(dev_data,configBert)
    train(configBert,BertModel,train_iter,dev_iter)

