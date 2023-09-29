import torch
import torch.nn.functional as F
from torch import nn

hidden_size = 64


class RNN(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, output_size=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size,
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
