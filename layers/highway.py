import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class SeqLinear(nn.Module):

    def __init__(self, in_features, out_features):
        self.__in_features
        self.__out_features
        self.weight = Parameter(
            torch.Tensor(out_features, in_features))
        self.bias = Parameter(
            torch.Tensor(out_features))
        self._reset_parameters()

    @property
    def in_features(self):
        return self.__in_features

    @property
    def out_features(self):
        return self.__out_features

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Args:
            x: A Tensor of size (batch_size, in_features, time_steps)
        """
        x = torch.bmm(self.weight, x)
        bias = torch.unsqueeze(torch.unsqueeze(bias, 0), 2)
        return x + bias.expand(x.size())

class HighwayNet(nn.Module):

    def __init__(self, num_layers, gate_fc_bias=-1):
        super(HighwayNet, self).__init__()
        self.__num_layers = num_layers
        self.__gate_fc_bias = gate_fc_bias

    @property
    def num_layers(self):
        return self.__num_layers

    @property
    def gate_fc_bias(self):
        return self.__gate_fc_bias

    def _single_layer(self, x):
        in_features = x.size()[1]
        out_features = in_features
        fc = SeqLinear(in_features, out_features)
        gate_fc = SeqLinear(in_features, out_features)
        gate_fc.bias.data.fill_(self.gate_fc_bias)

        h = F.relu(fc(x))
        t = F.softmax(gate_fc(x))
        return torch.cmul(h, t) + torch.cmul(x, (1 - t))

    def forward(self, x):
        """
        Args:
            x: A Tensor of size (batch_size, in_features, time_steps)
        """
        for l in range(self.num_layers):
            y = _single_layer(x)
            x = y
        return y
