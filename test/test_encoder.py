import numpy as np
from modules.encoder import *
from torch.autograd import Variable


def test_encoder():
    num_embeddings = 52  # suppose we only have letters
    embedding_dim = 256

    bank_k = 16
    bank_ck = 128
    proj_dims = (128, 128)
    highway_layers = 4
    highway_units = 128
    gru_units = 128

    batch_size = 32
    max_length = 30

    input = Variable(
        torch.LongTensor(
            np.random.randint(
                0,
                high=num_embeddings-1,
                size=(batch_size, max_length)
            )
        )
    )

    encoder = Encoder(num_embeddings, embedding_dim,
                      bank_k, bank_ck, proj_dims, highway_layers,
                      highway_units, gru_units)

    output = encoder(input)
    assert output.size() == (batch_size, max_length, 2 * gru_units)

    output[0, 0, 0].backward()
