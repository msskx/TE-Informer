import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, features_size=32, input_size=11, hidden_size=64, output_size=1):
        super(MLP, self).__init__()
        self.fc = nn.Linear(features_size * input_size,
                            hidden_size)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, inputs):
        inputs = inputs.to(torch.float)
        inputs = inputs.view(inputs.shape[0], -1)
        x = F.dropout(F.relu(self.fc(inputs)))
        x = F.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.squeeze(x)
