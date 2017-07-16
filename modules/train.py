import time
import math
import random
import sys
import traceback
import numpy as np
import torch
import argparse
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from modules.decoder import AttnDecoder
from modules.encoder import Encoder
from modules.postnet import PostNet
from modules.dataset import tiny_words
from modules.hyperparams import Hyperparams as hp
from utils import Timed


parser = argparse.ArgumentParser(
    description="Train an Tacotron model for speech synthesis")
parser.add_argument("--max-epochs", type=int, default=100000)
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true')
parser.set_defaults(use_cuda=False)
parser.add_argument('-d', '--data-size', default=sys.maxsize, type=int)


def train_batch(mels_v, mags_v, texts_v,
                encoder, decoder, postnet,
                optimizer, criterion, clip=5.0):
    """
    Args:
        texts_v: A Tensor of size (batch_size, max_text_length)
        mels_v: A Tensor of size
            (batch_size, max_audio_length, frame_size)
        mags_v: A Tensor of size (batch_size, max_audio_length, ???)
    """

    # zero gradients
    optimizer.zero_grad()
    # added onto for each frame
    loss = 0

    # get batch size and initialize GO_frame
    GO_frame = np.zeros((hp.batch_size, hp.n_mels))

    # get target length
    T = hp.max_audio_length

    # encoder
    encoder_out = encoder(texts_v)

    # Prepare input and output variables
    decoder_in = Variable(torch.from_numpy(GO_frame).float())
    if hp.use_cuda:
        decoder_in = decoder_in.cuda()
    h, hs = decoder.init_hiddens(hp.batch_size)

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < hp.teacher_forcing_ratio
    decoder_outs = []
    if use_teacher_forcing:

        # Teacher forcing: Use the ground-truth target as the next input
        for t in range(int(T / hp.rf)):
            # decoder
            # decoder_out: (batch_size, hp.rf, hp.n_mels)
            decoder_out, h, hs, _ = decoder(decoder_in, h, hs, encoder_out)
            decoder_outs.append(decoder_out)

            mel_truth = mels_v[:, hp.rf*t: hp.rf*(t+1), :]
            loss += criterion(decoder_out, mel_truth)
            # use truth
            decoder_in = mels_v[:, hp.rf*(t+1)-1, :].contiguous()

    else:
        # Without teacher forcing: use network's prediction as the next input
        for t in range(int(T / hp.rf)):
            # decoder
            # decoder_out: (batch_size, hp.rf, hp.n_mels)
            decoder_out, h, hs, _ = decoder(decoder_in, h, hs, encoder_out)
            decoder_outs.append(decoder_out)

            mel_truth = mels_v[:, hp.rf*t: hp.rf*(t+1), :]
            loss += criterion(decoder_out, mel_truth)
            # use predict
            decoder_in = decoder_out[:, -1, :].contiguous()

    # (batch_size, T, n_mels)
    decoder_outs = torch.cat(decoder_outs, 1)

    # postnet
    post_out = postnet(decoder_outs)
    loss += criterion(post_out, mags_v)

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    optimizer.step()

    return loss.data[0] / T


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def train(args):
    # initalize dataset
    with Timed('Loading dataset'):
        ds = tiny_words(
            max_text_length=hp.max_text_length,
            max_audio_length=hp.max_audio_length,
            max_dataset_size=args.data_size
        )

    # initialize model
    with Timed('Initializing model.'):
        encoder = Encoder(
            ds.lang.num_chars, hp.embedding_dim, hp.encoder_bank_k,
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

        if hp.use_cuda:
            encoder.cuda()
            decoder.cuda()
            postnet.cuda()

        # initialize optimizers and criterion
        all_paramters = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.Adam(all_paramters, lr=hp.lr)
        criterion = nn.L1Loss()

        # configuring traingin
        plot_every = 200
        print_every = 100

        # Keep track of time elapsed and running averages
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

    for epoch in range(1, hp.n_epochs + 1):

        # get training data for this cycle
        mels, mags, indexed_texts = ds.next_batch(hp.batch_size)

        mels_v = Variable(torch.from_numpy(mels).float())
        mags_v = Variable(torch.from_numpy(mags).float())
        texts_v = Variable(torch.from_numpy(indexed_texts))

        if hp.use_cuda:
            mels_v = mels_v.cuda()
            mags_v = mags_v.cuda()
            texts_v = texts_v.cuda()

        loss = train_batch(
            mels_v, mags_v, texts_v,
            encoder, decoder, postnet,
            optimizer, criterion
        )

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0:
            continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % \
                (time_since(start, epoch / hp.n_epochs),
                 epoch, epoch / hp.n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def main():
    args = parser.parse_args()

    try:
        return train(args)
    except Exception as e:
        traceback.print_exc()
        print('[Error]', str(e))
        return 1

if __name__ == "__main__":
    exit(main())
