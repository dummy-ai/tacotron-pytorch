from modules.conv1d import *
from torch.autograd import Variable

def test_conv1d():
    k = 16
    ck = 128
    conv1dbank = Conv1dBankWithMaxPool(k, ck)

    batch_size = 32
    in_channels = 256
    time_steps = 10
    inp = Variable(torch.randn(batch_size, in_channels, time_steps))

    out = conv1dbank(inp)
    assert out.size() == (batch_size, k * ck, time_steps)

    proj_dims = (256, 80)
    conv1dproj = Conv1dProjection(proj_dims)
    final_out = conv1dproj(out)
    assert(final_out.size() == 
        (batch_size, proj_dims[1], time_steps)) 








