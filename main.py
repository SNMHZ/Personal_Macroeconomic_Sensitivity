import os
import torch
from dataset import Preprocessor, Preprocessor_MR
from model import BaseLine, MultiRNNModel
from train import Trainer_MR

def main():
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_path = os.path.join( os.getcwd(), 'dataset/train/df19_train.parquet')
    validation_path = os.path.join( os.getcwd(), 'dataset/train/df19_train.parquet')

    preprocessor = Preprocessor_MR(train_path)
    inputs, label = preprocessor.load_and_preprocess(train_path)
    inputs_test, label_test = preprocessor.load_and_preprocess(validation_path)
    
    model = MultiRNNModel()
    trainer_MR = Trainer_MR(model, torch_device)

    loss_list, test_loss_list = trainer_MR.train(inputs, inputs_test, label, label_test, 100)

if __name__=='__main__':
    main()