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
parser.add_argument('--multi-gpus', dest='multi_gpus', default=False, action='store_true')
parser.add_argument('-d', '--data-size', default=sys.maxsize, type=int)


def train_batch(mels_v, mags_v, texts_v, postnet,
                optimizer, criterion,
                multi_gpus=False, clip=5.0):
    """
    Args:
        texts_v: A Tensor of size (batch_size, max_text_length)
        mels_v: A Tensor of size
            (batch_size, max_audio_length, frame_size)
        mags_v: A Tensor of size (batch_size, max_audio_length, ???)
    """

    # zero gradients
    optimizer.zero_grad()

    # get target length
    T = hp.max_audio_length

    # postnet
    post_out = postnet(mels_v)
    loss = criterion(post_out, mags_v)

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(postnet.parameters(), clip)
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
        postnet = PostNet(
            hp.n_mels, 1 + hp.n_fft//2,
            hp.post_bank_k, hp.post_bank_ck,
            hp.post_proj_dims, hp.post_highway_layers, hp.post_highway_units,
            hp.post_gru_units, use_cuda=hp.use_cuda
        )

        if args.multi_gpus:
            all_devices = list(range(torch.cuda.device_count()))
            postnet = nn.DataParallel(postnet, device_ids=all_devices)

        if hp.use_cuda:
            postnet.cuda()

        # initialize optimizers and criterion
        all_paramters = (list(postnet.parameters()))
        optimizer = optim.Adam(all_paramters, lr=hp.lr)
        criterion = nn.L1Loss()

        # configuring traingin
        print_every = 100
        save_every = 1000

        # Keep track of time elapsed and running averages
        start = time.time()
        print_loss_total = 0  # Reset every print_every

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
            mels_v, mags_v, texts_v, postnet,
            optimizer, criterion, multi_gpus=args.multi_gpus
        )

        # Keep track of loss
        print_loss_total += loss

        if epoch == 0:
            continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % \
                (time_since(start, epoch / hp.n_epochs),
                 epoch, epoch / hp.n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'postnet': postnet.state_dict(),
                'optimizer': optimizer.state_dict(),
            })


def save_checkpoint(state, filename="tacotron.checkpoint"):
    torch.save(state, filename)


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
