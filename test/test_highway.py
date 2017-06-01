from modules.highway import *
from torch.autograd import Variable

def test_seqlinear():
    batch_size = 32
    in_channels = 128 * 16
    time_steps = 15 
    inp = Variable(torch.ones(batch_size, in_channels, time_steps))

    out_features = 128
    seqlinear = SeqLinear(in_channels, out_features)

    out = seqlinear(inp)
    assert out.size() == (batch_size, out_features, time_steps)

def test_highwaynet():
    batch_size = 32
    in_channels = 80
    time_steps = 15
    inp = Variable(torch.ones(batch_size, in_channels, time_steps))

    num_layers = 4
    num_units = 128
    highway = HighwayNet(4, 128)

    out = highway(inp)
    assert out.size() == (batch_size, num_units, time_steps)



