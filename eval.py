import torch
import torch.nn as nn
import argparse
import sys
import numpy as np
from torch.autograd import Variable
from modules.decoder import AttnDecoder
from modules.encoder import Encoder
from modules.postnet import PostNet
from modules.dataset import make_lang, TINY_WORDS, indexes_from_text, pad_indexes
from modules.audio_signal import spectrogram2wav, griffinlim
from modules.hyperparams import Hyperparams as hp
from scipy.io.wavfile import write

EOT_token = 0
PAD_token = 1

parser = argparse.ArgumentParser(
    description="Generate wav based on given text")
parser.add_argument("--checkpoint", type=str, default="tacotron.checkpoint")
parser.add_argument("--text", type=str, default="hello")
parser.add_argument('-d', '--data-size', default=sys.maxsize, type=int)
args = parser.parse_args()


def inference(checkpoint_file, text):
    lang = make_lang(TINY_WORDS)

    print('Num characters', lang.num_chars)

    # prepare input
    indexes = indexes_from_text(lang, text)
    indexes.append(EOT_token)
    padded_indexes = pad_indexes(indexes, hp.max_text_length, PAD_token)
    texts_v = Variable(torch.from_numpy(padded_indexes))
    texts_v = texts_v.unsqueeze(0)

    if hp.use_cuda:
        texts_v = texts_v.cuda()

    encoder = Encoder(
        lang.num_chars, hp.embedding_dim, hp.encoder_bank_k,
        hp.encoder_bank_ck, hp.encoder_proj_dims,
        hp.encoder_highway_layers, hp.encoder_highway_units,
        hp.encoder_gru_units, dropout=hp.dropout, use_cuda=hp.use_cuda
    )

    decoder = AttnDecoder(
        hp.max_text_length, hp.attn_gru_hidden_size, hp.n_mels,
        hp.rf, hp.decoder_gru_hidden_size,
        hp.decoder_gru_layers,
        dropout=hp.dropout, use_cuda=hp.use_cuda
    )

    postnet = PostNet(
        hp.n_mels, 1 + hp.n_fft//2,
        hp.post_bank_k, hp.post_bank_ck,
        hp.post_proj_dims, hp.post_highway_layers, hp.post_highway_units,
        hp.post_gru_units, use_cuda=hp.use_cuda
    )

    encoder.eval()
    decoder.eval()
    postnet.eval()

    if hp.use_cuda:
        encoder.cuda()
        decoder.cuda()
        postnet.cuda()

    # load model
    checkpoint = torch.load(checkpoint_file)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    postnet.load_state_dict(checkpoint['postnet'])

    encoder_out = encoder(texts_v)

    # Prepare input and output variables
    GO_frame = np.zeros((1, hp.n_mels))
    decoder_in = Variable(torch.from_numpy(GO_frame).float())
    if hp.use_cuda:
        decoder_in = decoder_in.cuda()
    h, hs = decoder.init_hiddens(1)

    decoder_outs = []
    for t in range(int(hp.max_audio_length / hp.rf)):
        decoder_out, h, hs, _ = decoder(decoder_in, h, hs, encoder_out)
        decoder_outs.append(decoder_out)
        # use predict
        decoder_in = decoder_out[:, -1, :].contiguous()

    # (batch_size, T, n_mels)
    decoder_outs = torch.cat(decoder_outs, 1)

    # postnet
    post_out = postnet(decoder_outs)
    s = post_out[0].cpu().data.numpy()

    print("Recontructing wav...")
    s = np.where(s < 0, 0, s)
    wav = spectrogram2wav(s**hp.power)
    # wav = griffinlim(s**hp.power)
    write("demo.wav", hp.sr, wav)


def main():
    inference(args.checkpoint, args.text)


if __name__ == "__main__":
    main()
