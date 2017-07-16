import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.commons import SeqLinear
from modules.cbhg import CBHG


class PostNet(nn.Module):

    def __init__(self, in_channels, out_channels,
                 bank_k, bank_ck, proj_dims,
                 highway_layers, highway_units,
                 gru_units, use_cuda=False):
        super(PostNet, self).__init__()
        self.cbhg = CBHG(
            in_channels, bank_k, bank_ck,
            proj_dims, highway_layers, highway_units,
            gru_units, use_cuda=use_cuda
        )
        self.out = SeqLinear(gru_units * 2, out_channels, time_dim=1)

    def forward(self, x):
        """
        Args:
            x: A Tensor of size (batch_size, T, in_channels)

        Returns:
            A Tensor of size (batch_size, T, out_channels)
        """
        return self.out(self.cbhg(x).contiguous())
