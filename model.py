import torch
import torch.nn as nn

class BaseLine(nn.Module):
    def __init__(self):
        super(BaseLine, self).__init__()
        feats1 = ['급여-2', '급여-1', '신용-2', '신용-1', '체크-2', '체크-1', '수신-2', '수신-1', '현금-2', '현금-1']
        feats2 = ['전문직여부','세분화고수신고객여부','세분화고소득고객여부', '실적기준고객우대구분코드', '방카슈랑스보유좌수', '수익증권좌수', '신탁좌수', '연간상품가입건수', '전월대출월평균잔액', '급여이체여부',
'자동이체거래건수', '수신좌수']
        input_size = len(feats1)+len(feats2)

        toPredicts = ['신용스프레드민감도', '위안화민감도', '달러민감도', '실업률민감도', '비제조업민감도', '제조업민감도']
        toPredict_size = len(toPredicts)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_size, int(input_size*1.5) ),
            nn.ReLU(),
            nn.Linear(int(input_size*1.5), int(input_size*1.5)),
            nn.ReLU(),
            nn.Linear(int(input_size*1.5), int(input_size*1.5)),
            nn.ReLU(),
            nn.Linear(int(input_size*1.5), input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, toPredict_size),
            nn.Sigmoid()
        )

    def forward(self, data):
        return self.classifier(data)

class MultiRNNModel(nn.Module):
    def __init__(self):
        super(MultiRNNModel, self).__init__()
        toPredicts = ['신용스프레드민감도', '위안화민감도', '달러민감도', '실업률민감도', '비제조업민감도', '제조업민감도']
        seq_len = 2
        toPredict_size = len(toPredicts)

        self.lstm_layers = [ nn.LSTM(input_size=seq_len, 
                                    hidden_size=seq_len // 2,
                                    batch_first=True,bidirectional=False) for _ in range(toPredict_size)]
        
        self.classifier = nn.Sequential(
            nn.Linear(toPredict_size, toPredict_size), 
            nn.Sigmoid(),
            nn.Linear(toPredict_size, toPredict_size)
        )
    
    def forward(self, data):
        outputs = self.classifier(data)
        return outputs