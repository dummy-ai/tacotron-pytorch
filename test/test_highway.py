from modules.highway import *
from torch.autograd import Variable
import numpy.random as npr
import numpy as np

def test_seqlinear():
    batch_size = 32
    in_channels = 128 * 16
    time_steps = 15
    inp = Variable(torch.ones(batch_size, in_channels, time_steps))

    out_features = 128
    seqlinear = SeqLinear(in_channels, out_features)

    out = seqlinear(inp)
    assert out.size() == (batch_size, out_features, time_steps)

def test_seqlinear_batch():
    batch_size = 32
    in_channels = 128 * 16
    time_steps = 15
    input1 = np.array(npr.randn(1, in_channels, time_steps), dtype='float32')
    input1_single = Variable(torch.from_numpy(input1))
    input1_batch = Variable(torch.from_numpy(np.repeat(input1, batch_size, 0)))

    out_features = 128
    seqlinear = SeqLinear(in_channels, out_features)

    output1_single = seqlinear(input1_single).data.numpy()
    output1_batch = seqlinear(input1_batch).data.numpy()

    assert np.mean(np.abs(np.repeat(output1_single, batch_size, 0) - output1_batch)) < 1e-3

def test_highwaynet():
    batch_size = 32
    in_channels = 80
    time_steps = 15
    inp = Variable(torch.ones(batch_size, in_channels, time_steps))

    num_layers = 4
    num_units = 128
    highway = HighwayNet(in_channels, num_layers, num_units)

    out = highway(inp)
    assert out.size() == (batch_size, num_units, time_steps)


def test_highwaynet_batch():
    batch_size = 32
    in_channels = 80
    time_steps = 15
    input1 = np.array(npr.randn(1, in_channels, time_steps), dtype='float32')
    input1_single = Variable(torch.from_numpy(input1))
    input1_batch = Variable(torch.from_numpy(np.repeat(input1, batch_size, 0)))

    num_layers = 4
    num_units = 128
    highway = HighwayNet(in_channels, num_layers, num_units)

    output1_single = highway(input1_single).data.numpy()
    output1_batch = highway(input1_batch).data.numpy()

    assert np.mean(np.abs(np.repeat(output1_single, batch_size, 0) - output1_batch)) < 1e-3



