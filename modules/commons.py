import torch
import torch.nn as nn
import torch.nn.functional as F


def _wx(w, x):
    return torch.bmm(w.unsqueeze(0).expand(
        x.size()[0], w.size()[0], w.size()[1]), x)


class SeqLinear(nn.Module):
    """A wrapper around nn.Linear to handle 3D tensor"""

    def __init__(self, in_features, out_features, time_dim=2):
        super(SeqLinear, self).__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._time_dim = time_dim
        self.linear = nn.Linear(in_features, out_features)

    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_features

    @property
    def time_dim(self):
        return self._time_dim

    def forward(self, x):
        """
        Args:
            x: If self.time_dim = 2, a Tensor of size
                (batch_size, in_features, time_steps)
               elif self.time_dim = 1, a Tensor of size
                (batch_size, time_steps, in_features)
        Returns:
            If self.time_dim = 2, a Tensor of size
                (batch_size, out_features, time_steps)
            elif self.time_dim = 1, a Tensor of size
                (batch_size, time_steps, out_features)
        """
        batch_size = x.size()[0]
        if self.time_dim == 2:
            x = x.transpose(1, 2).contiguous()

        x = x.view(-1, self.in_features)

        # out has size (batch_size, time_steps, out_features)
        out = self.linear(x).view(batch_size, -1, self.out_features)

        # switch back feature and time dimension
        if self.time_dim == 2:
            out = out.contiguous().transpose(1, 2)

        return out
