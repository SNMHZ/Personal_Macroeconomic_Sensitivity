import torch
import torch.nn as nn

class BaseLine(nn.Module):
    def __init__(self):
        super(BaseLine, self).__init__()
        input_size = 22
        toPredict_size = 6

        self.classifier = nn.Sequential(
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
                                    hidden_size=seq_len // 2) for _ in range(toPredict_size)]

        self.classifier = nn.Sequential(
            nn.Linear(toPredict_size, toPredict_size), 
            nn.ReLU(),
            nn.Linear(toPredict_size, toPredict_size),
            nn.Sigmoid()
        )

    def forward(self, timeSeries, data):
        lstm_outputs = [lstm_layer(p) for lstm_layer, p in zip(self.lstm_layers, timeSeries)]
        lstm_data = torch.cat(lstm_outputs, data)
        outputs = self.classifier(lstm_data)
        return outputs