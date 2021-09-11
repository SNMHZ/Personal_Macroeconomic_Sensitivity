import numpy as np
import pandas as pd
import torch

class Preprocessor():
    def __init__(self, base_dataset_file_path):
        self.base_dataset = pd.read_parquet(base_dataset_file_path)
        self.base_dataset = self.makeTimeSeries(self.base_dataset)
        
        self.feats1 = ['급여-2', '급여-1', '신용-2', '신용-1', '체크-2', '체크-1', '수신-2', '수신-1', '현금-2', '현금-1']
        self.feats2 = ['전문직여부', '세분화고수신고객여부', '세분화고소득고객여부', '실적기준고객우대구분코드', 
                    '방카슈랑스보유좌수', '수익증권좌수', '신탁좌수', 
                    '연간상품가입건수', '전월대출월평균잔액', '급여이체여부', '자동이체거래건수', '수신좌수']
        self.labels = ['신용스프레드민감도', '위안화민감도', '달러민감도', '실업률민감도', '비제조업민감도', '제조업민감도']

        self.scaleVector = self.base_dataset[self.feats1 + self.feats2].max().values
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def makeTimeSeries(self, df: pd.DataFrame):
        df['급여-2']=(df['삼개월급여이체실적금액']-df['전월급여이체실적금액']-df['급여이체실적금액'])
        df['급여-2'].mask(df['급여-2']<0, 0, inplace=True)
        df['급여-1']=df['전월급여이체실적금액']
        df['급여-0']=df['급여이체실적금액']

        df['신용-2']=(df['삼개월신용카드사용금액']-df['전월신용카드사용금액']-df['신용카드사용금액'])
        df['신용-2'].mask(df['신용-2']<0, 0, inplace=True)
        df['신용-1']=df['전월신용카드사용금액']
        df['신용-0']=df['신용카드사용금액']

        df['체크-2']=(df['삼개월체크카드금액']-df['전월체크카드금액']-df['체크카드거래금액'])
        df['체크-2'].mask(df['체크-2']<0, 0, inplace=True)
        df['체크-1']=df['전월체크카드금액']
        df['체크-0']=df['체크카드거래금액']

        df['수신-2']=(df['삼개월수신평균잔액']*3-df['전월수신평균잔액']-df['수신잔액'])
        df['수신-2'].mask(df['수신-2']<0, 0, inplace=True)
        df['수신-1']=df['전월수신평균잔액']
        df['수신-0']=df['수신잔액']

        df['현금-2']=(df['삼개월현금서비스금액']-df['전월현금서비스금액']-df['현금서비스이용금액'])
        df['현금-2'].mask(df['현금-2']<0, 0, inplace=True)
        df['현금-1']=df['전월현금서비스금액']
        df['현금-0']=df['현금서비스이용금액']

        df['대출-2']=(df['삼개월대출평균잔액']*3-df['전월대출월평균잔액']-df['가계자금대출잔액']-df['주택담보대출잔액'])
        df['대출-2'].mask(df['대출-2']<0, 0, inplace=True)
        df['대출-1']=df['전월대출월평균잔액']
        df['대출-0']=df['가계자금대출잔액']+df['주택담보대출잔액']
        return df

    def scaleInputData(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.feats1 + self.feats2] / self.scaleVector

    def load_and_preprocess(self, dataset_file_path):
        dataset = pd.read_parquet(dataset_file_path)
        dataset = self.makeTimeSeries(dataset)
        
        input_data = self.scaleInputData(dataset).values
        label_data = dataset[self.labels].values

        return torch.as_tensor(input_data, dtype=torch.float32, device=self.device), torch.as_tensor(label_data, dtype=torch.float32, device=self.device)

class Preprocessor_MR():
    def __init__(self, base_dataset_file_path):
        self.base_dataset = pd.read_parquet(base_dataset_file_path)
        self.base_dataset = self.makeTimeSeries(self.base_dataset)
        
        self.feats1 = ['급여-2', '급여-1', '신용-2', '신용-1', '체크-2', '체크-1', '수신-2', '수신-1', '현금-2', '현금-1', '대출-2', '대출-1']
        self.feats2 = ['방카슈랑스보유좌수', '수익증권좌수', '신탁좌수', '연간상품가입건수', '전월대출월평균잔액', '자동이체거래건수', '수신좌수']
        self.feats3 = ['전문직여부', '세분화고수신고객여부', '세분화고소득고객여부', '실적기준고객우대구분코드', '급여이체여부']
        
        self.labels = ['원달러_민감도', 'CPI_민감도', '제조업_민감도', '비제조업_민감도', '고용율_민감도']

        for col in self.feats1+self.feats2:
          self.winsorizeOutlier(self.base_dataset[col])
        
        self.scaleVector = self.base_dataset[self.feats1 + self.feats2 + self.feats3].max().values
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def makeTimeSeries(self, df: pd.DataFrame):
        df['급여-2']=(df['삼개월급여이체실적금액']-df['전월급여이체실적금액']-df['급여이체실적금액'])
        df['급여-2'].mask(df['급여-2']<0, 0, inplace=True)
        df['급여-1']=df['전월급여이체실적금액']
        df['급여-0']=df['급여이체실적금액']

        df['신용-2']=(df['삼개월신용카드사용금액']-df['전월신용카드사용금액']-df['신용카드사용금액'])
        df['신용-2'].mask(df['신용-2']<0, 0, inplace=True)
        df['신용-1']=df['전월신용카드사용금액']
        df['신용-0']=df['신용카드사용금액']

        df['체크-2']=(df['삼개월체크카드금액']-df['전월체크카드금액']-df['체크카드거래금액'])
        df['체크-2'].mask(df['체크-2']<0, 0, inplace=True)
        df['체크-1']=df['전월체크카드금액']
        df['체크-0']=df['체크카드거래금액']

        df['수신-2']=(df['삼개월수신평균잔액']*3-df['전월수신평균잔액']-df['수신잔액'])
        df['수신-2'].mask(df['수신-2']<0, 0, inplace=True)
        df['수신-1']=df['전월수신평균잔액']
        df['수신-0']=df['수신잔액']

        df['현금-2']=(df['삼개월현금서비스금액']-df['전월현금서비스금액']-df['현금서비스이용금액'])
        df['현금-2'].mask(df['현금-2']<0, 0, inplace=True)
        df['현금-1']=df['전월현금서비스금액']
        df['현금-0']=df['현금서비스이용금액']

        df['대출-2']=(df['삼개월대출평균잔액']*3-df['전월대출월평균잔액']-df['가계자금대출잔액']-df['주택담보대출잔액'])
        df['대출-2'].mask(df['대출-2']<0, 0, inplace=True)
        df['대출-1']=df['전월대출월평균잔액']
        df['대출-0']=df['가계자금대출잔액']+df['주택담보대출잔액']
        return df

    def scaleInputData(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.feats1 + self.feats2 + self.feats3] / self.scaleVector
    
    def winsorizeOutlier(self, series: pd.Series):
        q1, q3 = np.percentile(series[series!=0], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)
        series.mask(series > upper_bound, upper_bound, inplace=True)
        series.mask(series < lower_bound, lower_bound, inplace=True)
      
    def as_tensor(self, data):
      return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def load_and_preprocess(self, dataset_file_path):
        dataset = pd.read_parquet(dataset_file_path)
        dataset = self.makeTimeSeries(dataset)

        for col in self.feats1 + self.feats2:
          self.winsorizeOutlier(dataset[col])

        input_data = self.scaleInputData(dataset).values
        seq0 = input_data[:, :2].reshape(1, input_data.shape[0], 2)
        seq1 = input_data[:, 2:4].reshape(1, input_data.shape[0], 2)
        seq2 = input_data[:, 4:6].reshape(1, input_data.shape[0], 2)
        seq3 = input_data[:, 6:8].reshape(1, input_data.shape[0], 2)
        seq4 = input_data[:, 8:10].reshape(1, input_data.shape[0], 2)
        others = input_data[:, 10:].reshape(1, input_data.shape[0], input_data.shape[1]-10)
        label_data = dataset[self.labels].values

        return ((self.as_tensor(seq0), self.as_tensor(seq1), self.as_tensor(seq2), self.as_tensor(seq3), self.as_tensor(seq4)), self.as_tensor(others)), self.as_tensor(label_data)