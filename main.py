import os
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from dataset import Preprocessor, Preprocessor_MR
from model import BaseLine, MultiRNNModel
from train import Trainer_MR


def main():
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_path = os.path.join( os.getcwd(), 'dataset/train/df19_train.parquet')
    validation_path = os.path.join( os.getcwd(), 'dataset/train/df19_train.parquet')

    batchSize=4096

    preprocessor = Preprocessor_MR(train_path)
    train_inputs, train_label = preprocessor.load_and_preprocess(train_path)
    train_dataset = TensorDataset(train_inputs, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, drop_last=False)

    valid_inputs, valid_label = preprocessor.load_and_preprocess(validation_path)
    valid_dataset = TensorDataset(valid_inputs, valid_label)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batchSize, shuffle=True, drop_last=False)
    
    model = MultiRNNModel()
    trainer_MR = Trainer_MR(model, torch_device)

    loss_list, test_loss_list = trainer_MR.train(train_dataloader, valid_dataloader, 100)

if __name__=='__main__':
    main()