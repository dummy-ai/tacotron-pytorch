from modules.decoder import *
from torch.autograd import Variable

def test_decoder():
    batch_size = 32
    attn_gru_hidden_size = 256
    decoder_gru_hidden_size = 256
    frame_size = 80
    num_frames = 3
    max_text_length = 30

    decoder = AttnDecoder(max_text_length)
    attn_gru_hidden, decoder_gru_hiddens = decoder.init_hiddens(batch_size)

    input = Variable(torch.randn(batch_size, frame_size)) 
    encoder_outputs = Variable(
        torch.randn(batch_size, max_text_length, attn_gru_hidden_size))

    output, new_attn_gru_hidden, new_decoder_gru_hiddens, attn_weights = \
        decoder(input, attn_gru_hidden, decoder_gru_hiddens, encoder_outputs)

    assert output.size() == (batch_size, frame_size * num_frames) 
    assert new_attn_gru_hidden.size() == (batch_size, attn_gru_hidden_size)
    assert attn_weights.size() == (batch_size, max_text_length)
    assert len(new_decoder_gru_hiddens) == 2
    assert new_decoder_gru_hiddens[0].size() == (batch_size, decoder_gru_hidden_size)

    decoder(input, new_attn_gru_hidden, new_decoder_gru_hiddens, encoder_outputs)
