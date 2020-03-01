import torch
import torch.nn as nn

class LSTMNet(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, bidirectional=False)

    def forward_once(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x.permute(1, 0, 2))
        return h_n.view(h_n.size(1), -1)

    def forward(self, text_one, text_two):
        lstm_out1 = self.forward_once(text_one)
        lstm_out2 = self.forward_once(text_two)

        return lstm_out1, lstm_out2