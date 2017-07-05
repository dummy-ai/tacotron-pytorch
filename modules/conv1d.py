import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv_helper(input, conv_layer, kernel_size):
    """If kernel_size is even, slice conv_out on the time dimension
    to preserve the original time resolution
    """
    if kernel_size % 2 == 0:
        return conv_layer(input)[:, :, :-1]
    else:
        return conv_layer(input)

class Conv1dBankWithMaxPool(nn.Module):

    def __init__(self, in_channels, k, ck,
                 max_pool_width=2,
                 activation=F.relu):
        super(Conv1dBankWithMaxPool, self).__init__()
        """
        Args:
            k: An int
            ck: An int
        """
        self._in_channels = in_channels
        self._k = k
        self._ck = ck
        self._max_pool_width = max_pool_width
        self._activation = activation

        self.norms = nn.ModuleList([nn.BatchNorm1d(ck) for i in range(k)])
        self.conv1ds = nn.ModuleList()
        for kernel_size in range(1, self.k + 1):
            padding = int(kernel_size / 2)
            """nn.Conv1d takes input of size (N, C, L)
            Args:
                ck: An integer specify out_channels
                stride: Use stride of 1 to preserve the original time resolution
            """
            self.conv1ds.append(
                nn.Conv1d(in_channels, ck, kernel_size,
                    stride=1 ,padding=padding
                )
            )

    @property
    def in_channels(self):
        return self._in_channels

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
        conv_lst = []
        for kernel_size, conv1d in enumerate(self.conv1ds, start=1):
            conv_out = _conv_helper(x, conv1d, kernel_size)
            norm_out = self.norms[kernel_size - 1](conv_out) 
            conv_lst.append(self.activation(norm_out))

        stacked_conv = torch.cat(conv_lst, dim=1) 

        # padding is on both sides
        pooled_conv = F.max_pool1d(stacked_conv, 
            self.max_pool_width, stride=1, padding=1) 

        # slice on the time dimension to preserve the
        # original time resolution
        return pooled_conv[:, :, :-1]

class Conv1dProjection(nn.Module):

    def __init__(self, in_channels, proj_dims,
        kernel_size=3, activation=F.relu):
        super(Conv1dProjection, self).__init__()
        """
        Args:
            kernel_size: An int
        """
        self._in_channels = in_channels
        self._proj_dims = proj_dims
        self._kernel_size = kernel_size
        self._activation = activation

        padding = int(kernel_size / 2)
        self.conv1 = nn.Conv1d(in_channels, proj_dims[0], kernel_size,
            stride=1, padding=padding) 
        self.conv2 = nn.Conv1d(proj_dims[0], proj_dims[1], kernel_size,
            stride=1, padding=padding) 
        self.norm1 = nn.BatchNorm1d(proj_dims[0])
        self.norm2 = nn.BatchNorm1d(proj_dims[1]) 

    @property
    def in_channels(self):
        return self._in_channels

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
                                 self.in_channels,
                                 time_steps)

        Returns:
            A Tensor of size (batch_size,
                              self.proj_dims[1],
                              time_steps)
        """
        conv1_out = _conv_helper(x, self.conv1, self.kernel_size)
        norm1_out = self.norm1(conv1_out) 
        act1_out = self.activation(norm1_out)
        conv2_out = _conv_helper(act1_out, self.conv2, self.kernel_size)
        norm2_out = self.norm2(conv2_out)
        return self.activation(norm2_out)





