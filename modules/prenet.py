import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.commons import SeqLinear

class PreNet(nn.Module):

    def __init__(self, input_size, fc1_hidden_size=256, fc2_hidden_size=128):
        super(PreNet, self).__init__()
        self._input_size = input_size
        self._fc1_hidden_size = fc1_hidden_size 
        self._fc2_hidden_size = fc2_hidden_size
        self.fc1 = SeqLinear(input_size, fc1_hidden_size, time_dim=1)
        self.fc2 = SeqLinear(fc1_hidden_size, fc2_hidden_size, time_dim=1)
        self.dropout = nn.Dropout(0.5)

    @property
    def input_size(self):
        return self._input_size

    @property
    def fc1_hidden_size(self):
        return self._fc1_hidden_size

    @property
    def fc2_hidden_size(self):
        return self._fc2_hidden_size

    def forward(self, x):
        """
        Args:
            x: A Tensor of size (batch_size, time_steps, input_size)

        Returns:
            A Tensor of size (batch_size, time_steps, fc2_hidden_size)
        """ 
        out1 = self.dropout(F.relu(self.fc1(x)))
        return self.dropout(F.relu(self.fc2(out1)))


