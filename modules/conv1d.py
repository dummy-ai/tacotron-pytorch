import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv1d(x, in_channels, out_channels, kernel_size):
    """Helper function to perform 1d convolution

    Use stride of 1 to preserve the original time resolution.

    Args:
        x: A Tensor of size (N, C, L)
        kernel_size: An int, if even, slice the output
            of convolution operation on the time dimension
            to preserve the orignal time resolution
    """
    # round down padding
    padding = int(kernel_size / 2)
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
        stride=1, padding=padding)

    if kernel_size % 2 == 0:
        return conv(x)[:, :, :-1] 
    else:
        return conv(x)

def _batchnorm1d(x, num_features):
    batchnorm = nn.BatchNorm1d(num_features)
    return batchnorm(x) 

class Conv1dBankWithMaxPool(nn.Module):

    def __init__(self, k, ck,
                 max_pool_width=2,
                 activation=F.relu):
        super(Conv1dBankWithMaxPool, self).__init__()
        """
        Args:
            k: An int
            ck: An int
        """

        self._k = k
        self._ck = ck
        self._max_pool_width = max_pool_width
        self._activation = activation

    @property
    def k(self):
        return self._k

    @property
    def ck(self):
        return self._ck

    @property
    def max_pool_width(self):
        return self._max_pool_width

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
        in_channels = x.size()[1]
        conv_lst = []
        for idk in range(self.k):
            conv_k = self.activation(
                        _conv1d(x, 
                                in_channels, 
                                self.ck, 
                                idk+1))

            norm_k = _batchnorm1d(conv_k, self.ck)
            conv_lst.append(norm_k)

        stacked_conv = torch.cat(conv_lst, dim=1) 

        # padding is on both sides
        pooled_conv = F.max_pool1d(stacked_conv, 
            self.max_pool_width, stride=1, padding=1) 

        # slice on the time dimension to preserve the
        # original time resolution
        return pooled_conv[:, :, :-1]

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
        """
        Args: 
            x: A Tensor of size (batch_size,
                                 in_channels,
                                 time_steps)

        Returns:
            A Tensor of size (batch_size,
                              self.proj_dims[1],
                              time_steps)
        """
        in_channels = x.size()[1]
        out1_channels = self.proj_dims[0]
        out2_channels = self.proj_dims[1] 

        out1 = self.activation(
                _conv1d(x, 
                    in_channels, 
                    out1_channels, 
                    self.kernel_size))
        norm_out1 = _batchnorm1d(out1, out1_channels)
        out2 = _conv1d(norm_out1, 
                    out1_channels, 
                    out2_channels,
                    self.kernel_size)
        norm_out2 = _batchnorm1d(out2, out2_channels)
        return norm_out2




