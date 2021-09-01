import os
from dataset import preprocessor

if __name__=='__main__':
    path = os.path.join( os.getcwd(), 'dataset/train/df19_train.parquet')
    preprocessor(path)