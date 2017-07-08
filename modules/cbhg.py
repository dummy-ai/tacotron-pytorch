import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules.conv1d import Conv1dBankWithMaxPool, Conv1dProjection
from modules.highway import HighwayNet

class CBHG(nn.Module):

    def __init__(self, in_channels, bank_k, bank_ck, proj_dims,
        highway_layers, highway_units, gru_units, gru_layers=1):
        super(CBHG, self).__init__()
        """
        Args:
            bank_k, bank_ck: Two integers for Conv1D Bank
                bank_k th set contains bank_ck filters of width k
            proj_dims: A pair of integers, specifying the
                projection dimensions in Conv1D projection layer
        """
        self._in_channels = in_channels
        self._bank_k = bank_k
        self._bank_ck = bank_ck
        self._proj_dims = proj_dims
        self._highway_layers = highway_layers
        self._highway_units = highway_units
        self._gru_units = gru_units
        self._gru_layers = gru_layers

        self.convbank = Conv1dBankWithMaxPool(in_channels, bank_k, bank_ck)
        proj_in_channels = bank_k * bank_ck
        self.convproj = Conv1dProjection(proj_in_channels, proj_dims)
        self.highway = HighwayNet(in_channels, highway_layers, highway_units)

        self.gru = nn.GRU(highway_units, gru_units, gru_layers,
                          batch_first=True,
                          bidirectional=True)

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def bank_k(self):
        return self._bank_k

    @property
    def bank_ck(self):
        return self._bank_ck

    @property
    def proj_dims(self):
        return self._proj_dims

    @property
    def highway_layers(self):
        return self._highway_layers

    @property
    def highway_units(self):
        return self._highway_units

    @property
    def gru_units(self):
        return self._gru_units

    @property
    def gru_layers(self):
        return self._gru_layers

    def forward(self, x):
        """
        Args:
            x: A Tensor of size (batch_size,
                                 time_steps,
                                 self.in_channels)

        Returns:
            A Tensor of size (batch_size,
                              time_steps,
                              2 * gru_units)
        """
        x = torch.transpose(x, 1, 2).contiguous()
        bank_out = self.convbank(x)
        proj_out = self.convproj(bank_out)
        res_out = x + proj_out
        highway_out = self.highway(res_out)

        # rnn_inputs has size (batch_size, time_steps, highway_units)
        rnn_inputs = torch.transpose(highway_out, 1, 2).contiguous()

        batch_size = x.size()[0]
        # TODO init_hidden
        h0 = Variable(torch.randn(self.gru_layers * 2,
                                  batch_size,
                                  self.gru_units))
        final_output, _ = self.gru(rnn_inputs, h0)
        return final_output
