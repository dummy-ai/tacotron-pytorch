from modules.cbhg import *
from torch.autograd import Variable

def test_cbhg():
    batch_size = 32
    # number of output features of pre-net
    in_channels = 128 
    time_steps = 15 
    inp = Variable(torch.ones(batch_size, time_steps, in_channels))

    bank_k = 16
    bank_ck = 128
    proj_dims = (128, 128)
    highway_layers = 4
    highway_units = 128
    gru_units = 128
    cbhg = CBHG(in_channels, bank_k, bank_ck, proj_dims, 
                highway_layers, highway_units, gru_units)

    out = cbhg(inp)
    assert out.size() == (batch_size, time_steps, 2 * gru_units)

