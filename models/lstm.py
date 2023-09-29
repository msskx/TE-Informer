import torch
from torch import nn
from torch.nn import functional as F
class LSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, output_size=1):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, inputs):
        inputs = inputs.to(torch.float)
        _, s_o = self.rnn(inputs)
        s_o = s_o[-1]
        x = F.dropout(F.relu(self.fc1(s_o)))
        x = self.fc2(x)
        return torch.squeeze(x)