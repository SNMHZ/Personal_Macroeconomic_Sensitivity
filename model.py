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
        seq_size = 11
        others_size = 12
        classifier_size = 34
        toPredict_size = 5

        self.lstm_layer0 = nn.LSTM(seq_len, seq_len)
        self.lstm_layer1 = nn.LSTM(seq_len, seq_len)
        self.lstm_layer2 = nn.LSTM(seq_len, seq_len)
        self.lstm_layer3 = nn.LSTM(seq_len, seq_len)
        self.lstm_layer4 = nn.LSTM(seq_len, seq_len)
        self.lstm_layer5 = nn.LSTM(seq_len, seq_len)

        self.lstm_layerm0 = nn.LSTM(seq_len, seq_len)
        self.lstm_layerm1 = nn.LSTM(seq_len, seq_len)
        self.lstm_layerm2 = nn.LSTM(seq_len, seq_len)
        self.lstm_layerm3 = nn.LSTM(seq_len, seq_len)
        self.lstm_layerm4 = nn.LSTM(seq_len, seq_len)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_size, int(classifier_size*1.5)), 
            nn.ReLU(),
            nn.Linear(int(classifier_size*1.5), int(classifier_size*1.5)), 
            nn.ReLU(),
            nn.Linear(int(classifier_size*1.5), classifier_size), 
            nn.ReLU(),
            nn.Linear(classifier_size, toPredict_size),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        seq0 = input_data[:, :2].reshape(1, input_data.shape[0], 2)
        seq1 = input_data[:, 2:4].reshape(1, input_data.shape[0], 2)
        seq2 = input_data[:, 4:6].reshape(1, input_data.shape[0], 2)
        seq3 = input_data[:, 6:8].reshape(1, input_data.shape[0], 2)
        seq4 = input_data[:, 8:10].reshape(1, input_data.shape[0], 2)
        seq5 = input_data[:, 10:12].reshape(1, input_data.shape[0], 2)

        m_seq0 = input_data[:, 12:14].reshape(1, input_data.shape[0], 2)
        m_seq1 = input_data[:, 14:16].reshape(1, input_data.shape[0], 2)
        m_seq2 = input_data[:, 16:18].reshape(1, input_data.shape[0], 2)
        m_seq3 = input_data[:, 18:20].reshape(1, input_data.shape[0], 2)
        m_seq4 = input_data[:, 20:22].reshape(1, input_data.shape[0], 2)

        others = input_data[:, 22:].reshape(1, input_data.shape[0], input_data.shape[1]-22)
        
        lstm_output0, _ = self.lstm_layer0(seq0)
        lstm_output1, _ = self.lstm_layer1(seq1)
        lstm_output2, _ = self.lstm_layer2(seq2)
        lstm_output3, _ = self.lstm_layer3(seq3)
        lstm_output4, _ = self.lstm_layer4(seq4)
        lstm_output5, _ = self.lstm_layer5(seq5)

        lstm_outputm0, _ = self.lstm_layerm0(m_seq0)
        lstm_outputm1, _ = self.lstm_layerm1(m_seq1)
        lstm_outputm2, _ = self.lstm_layerm2(m_seq2)
        lstm_outputm3, _ = self.lstm_layerm3(m_seq3)
        lstm_outputm4, _ = self.lstm_layerm4(m_seq4)

        cated = torch.cat([lstm_output0, lstm_output1, lstm_output2, lstm_output3, lstm_output4, lstm_output5, lstm_outputm0, lstm_outputm1, lstm_outputm2, lstm_outputm3, lstm_outputm4, others], dim=-1)
        cated = cated.reshape(cated.shape[1], cated.shape[2])
        outputs = self.classifier(cated)
        return outputs