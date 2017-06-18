import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from modules.commons import SeqLinear

class HighwayNet(nn.Module):

    def __init__(self, num_layers, num_units, gate_fc_bias=-1):
        super(HighwayNet, self).__init__()
        self._num_layers = num_layers
        self._gate_fc_bias = gate_fc_bias
        self._num_units = num_units

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def num_units(self):
        return self._num_units

    @property
    def gate_fc_bias(self):
        return self._gate_fc_bias

    def _single_layer(self, x):
        # initialize fc layers
        fc = SeqLinear(self.num_units, self.num_units)
        gate_fc = SeqLinear(self.num_units, self.num_units)
        gate_fc.linear.bias.data.fill_(self.gate_fc_bias)

        h1 = F.relu(fc(x))
        t = F.softmax(gate_fc(x))
        return h1 * t + x * (1-t) 

    def forward(self, x):
        """
        Args:
            x: A Tensor of size (batch_size, in_features, time_steps)

        Returns:
            A Tensor of size (batch_size, in_features, time_steps)
        """
        in_features = x.size()[1]
        if in_features != self.num_units:
            pre_fc = SeqLinear(in_features, self.num_units)
            x = pre_fc(x)
        for l in range(self.num_layers):
            out = self._single_layer(x)
            x = out
        return out
