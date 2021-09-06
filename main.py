import os
from dataset import Preprocessor
from model import BaseLine

if __name__=='__main__':
    train_path = os.path.join( os.getcwd(), 'dataset/train/df19_train.parquet')
    preprocessor = Preprocessor(train_path)
    preprocessor.load_and_preprocess(train_path)