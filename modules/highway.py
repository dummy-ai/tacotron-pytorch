import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from modules.commons import SeqLinear

class HighwayNet(nn.Module):

    def __init__(self, in_channels, num_layers, num_units, gate_fc_bias=-1.0):
        super(HighwayNet, self).__init__()
        self._in_channels = in_channels
        self._num_layers = num_layers
        self._gate_fc_bias = gate_fc_bias
        self._num_units = num_units

        if in_channels != num_units:
            self.pre_fc = SeqLinear(in_channels, num_units)

        # initialize fc layers
        self.Hs = nn.ModuleList(
            [SeqLinear(num_units, num_units) for i in range(num_layers)])

        # initalize gates
        self.Ts = nn.ModuleList()
        for i in range(num_layers):
            T = SeqLinear(num_units, num_units)
            T.linear.bias.data.fill_(self.gate_fc_bias)
            self.Ts.append(T)

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def num_units(self):
        return self._num_units

    @property
    def gate_fc_bias(self):
        return self._gate_fc_bias

    def forward(self, x):
        """
        Args:
            x: A Tensor of size (batch_size, in_channels, time_steps)

        Returns:
            A Tensor of size (batch_size, in_channels, time_steps)
        """
        if self.in_channels != self.num_units:
            x = self.pre_fc(x)
        for i in range(self.num_layers):
            h = F.relu(self.Hs[i](x))
            t = F.softmax(self.Ts[i](x))
            out = h * t + x * (1-t)
            x = out
        return out
