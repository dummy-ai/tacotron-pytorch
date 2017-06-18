from modules.decoder import *
from torch.autograd import Variable

def test_decoder():
    batch_size = 32
    attn_input_size = 128
    attn_hidden_size = 256
    decoder_output_size = 240 
    decoder_hidden_size = 256
    max_length = 30

    decoder = AttnDecoder()
    attn_hidden, decoder_hiddens = decoder.init_hiddens(batch_size)

    input = Variable(torch.randn(batch_size, attn_input_size))
    encoder_outputs = Variable(
        torch.randn(batch_size, max_length, attn_hidden_size))

    output, new_attn_hidden, new_decoder_hiddens, attn_weights = \
        decoder(input, attn_hidden, decoder_hiddens, encoder_outputs)

    assert output.size() == (batch_size, decoder_output_size)
    assert new_attn_hidden.size() == (batch_size, attn_hidden_size)
    assert attn_weights.size() == (batch_size, max_length)
    assert len(new_decoder_hiddens) == 2
    assert new_decoder_hiddens[0].size() == (batch_size, decoder_hidden_size)

    decoder(input, new_attn_hidden, new_decoder_hiddens, encoder_outputs)
