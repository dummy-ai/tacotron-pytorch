from modules.prenet import *
from torch.autograd import Variable
import numpy.random as npr
import numpy as np

def test_prenet():
    fc1_hidden_size = 256
    fc2_hidden_size = 128

    # simulate pre-net in decoder
    batch_size = 32
    input_size = 80
    input = Variable(torch.randn(batch_size, 1, input_size))

    prenet = PreNet(input_size,
                    fc1_hidden_size=fc1_hidden_size,
                    fc2_hidden_size=fc2_hidden_size)
    output = prenet(input)

    assert output.size() == (batch_size, 1, fc2_hidden_size)

    # simulate pre-net in encoder
    batch_size = 32
    embedding_size = 256
    time_steps = 17
    input2 = Variable(torch.randn(batch_size, time_steps, embedding_size))

    prenet2 = PreNet(embedding_size,
                     fc1_hidden_size=fc1_hidden_size,
                     fc2_hidden_size=fc2_hidden_size)

    output2 = prenet2(input2)

    assert output2.size() == (batch_size, time_steps, fc2_hidden_size)

def test_prenet_batch():
    fc1_hidden_size = 256
    fc2_hidden_size = 128

    # simulate pre-net in encoder
    batch_size = 32
    embedding_size = 256
    time_steps = 17
    input1 = np.array(npr.randn(1, time_steps, embedding_size), dtype='float32')
    input1_single = Variable(torch.from_numpy(input1))
    input1_batch = Variable(torch.from_numpy(np.repeat(input1, batch_size, 0)))

    prenet1 = PreNet(embedding_size,
                     fc1_hidden_size=fc1_hidden_size,
                     fc2_hidden_size=fc2_hidden_size)

    output1_single = prenet1(input1_single).data.numpy()
    output1_batch = prenet1(input1_batch).data.numpy()

    assert np.mean(np.abs(np.repeat(output1_single, batch_size, 0) - output1_batch)) < 1e-3


