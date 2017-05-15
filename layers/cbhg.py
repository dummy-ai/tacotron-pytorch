import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.conv1d import Conv1dBankWithMaxPool, Conv1dProjection
from layers.highway import HighwayNet

class CBHG(nn.Module):

    def __init__(self, bank_k, bank_ck, proj_dims):
        super(CBHG, self).__init__()

        self.__bank_k = bank_k
        self.__bank_ck = bank_ck
        self.__proj_dims = proj_dims

        self.convbank = Conv1dBankWithMaxPool(self.bank_k, self.bank_ck)
        self.convproj = Conv1dProjection()
        self.highway = HighwayNet(4)

    @property
    def bank_k(self):
        return self.__bank_k

    @property
    def bank_ck(self):
        return self.__bank_ck

    @property
    def proj_dims(self):
        return self.__proj_dims

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        bank_out = self.convbank(x)
        proj_out = self.convproj(x)
        res_out = x + proj_out
        highway_out = self.highway(res_out)

        hidden_size = 128
        self.rnn = nn.GRU(128, hidden_size, 1, 
                          batch_first=True,
                          bidirectional=True)
        rnn_inputs = torch.transpose(highway_out, 1, 2)

        batch_size = x.size()[0]
        h0 = Variable(torch.randn(2, batch_size, hidden_size))
        final_output, _ = self.rnn(rnn_inputs, h0)
        return final_output
