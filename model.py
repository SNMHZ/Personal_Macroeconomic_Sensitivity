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
        seq_len = 2
        seq_size = 5
        others_size = 14
        classifier_size = 24
        toPredict_size = 5

        self.lstm_layer0 = nn.LSTM(seq_len, seq_len)
        self.lstm_layer1 = nn.LSTM(seq_len, seq_len)
        self.lstm_layer2 = nn.LSTM(seq_len, seq_len)
        self.lstm_layer3 = nn.LSTM(seq_len, seq_len)
        self.lstm_layer4 = nn.LSTM(seq_len, seq_len)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_size, classifier_size), 
            nn.ReLU(),
            nn.Linear(classifier_size, toPredict_size),
            nn.Sigmoid()
        )

    def forward(self, seqs, others):
        lstm_output0, _ = self.lstm_layer0(seqs[0])
        lstm_output1, _ = self.lstm_layer1(seqs[1])
        lstm_output2, _ = self.lstm_layer2(seqs[2])
        lstm_output3, _ = self.lstm_layer3(seqs[3])
        lstm_output4, _ = self.lstm_layer4(seqs[4])

        cated = torch.cat([lstm_output0, lstm_output1, lstm_output2, lstm_output3, lstm_output4, others], dim=-1)
        cated = cated.reshape(cated.shape[1], cated.shape[2])
        outputs = self.classifier(cated)
        return outputs