import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv1d(x, in_channels, out_channels, width):
    conv = nn.Conv1d(in_channels, out_channels, width, stride=1)
    return conv(x)

def _batchnorm1d(x, num_features):
    batchnorm = nn.BatchNorm1d(num_features)
    return batchnorm(x) 

class Conv1dBankWithMaxPool(nn.Module):

    def __init__(self, k, ck, activation=F.relu):
        super(Conv1dBankWithMaxPool, self).__init__()
        """
        Args:
            k: An int
            ck: An int
        """

        self._k = k
        self._ck = ck
        self._activation = activation

    @property
    def k(self):
        return self._k

    @property
    def ck(self):
        return self._ck

    @property
    def activation(self):
        return self._activation

    def forward(self, x):
        """
        Args:
            x: A Tensor of size (batch_size, 
                                 embedding_size/in_channels,
                                 time_step/width)
        Returns:
            A Tensor of size (batch_size, 
                              k * ck,
                              time_step/width)

        """

        conv_lst = []
        for idk in range(self.k):
            conv_k = self.activation(
                        _conv1d(x, 
                                in_channels, 
                                ck, 
                                idk+1))
            norm_k = _batchnorm1d(x, ck)
            conv_lst.append(norm_k)

        stacked_conv = torch.stack(conv_list, axis=1) 

        pooled_conv = F.max_pool1d(stacked_conv, 2, 
                                    stride=1)

        return pooled_conv

class Conv1dProjection(nn.Module):

    def __init__(self, proj_dims, kernel_size=3, activation=F.relu):
        super(Conv1dProjection, self).__init__()
        """
        Args:
            kernel_size: An int
        """
        self._proj_dims = proj_dims
        self._kernel_size = kernel_size
        self._activation = activation

    @property
    def proj_dims(self):
        return self._proj_dims

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def activation(self):
        return self._activation

    def forward(self, x):
        in_channels = x.size()[1]
        out_channels1 = self.proj_dims[0]
        out_channels2 = self.proj_dims[1] 
        x = self.activation(
                _conv1d(x, 
                    in_channels, 
                    out_channels1, 
                    self.kernel_size))
        x = _batchnorm1d(x, out_channels1)
        x = _conv1d(x, 
                    out_channels1, 
                    out_channels2,
                    self.kernel_size)
        x = _batchnorm1d(x, out_channels2)
        return x




