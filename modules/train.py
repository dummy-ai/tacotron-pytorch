import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from modules.decoder import AttnDecoder
from modules.encoder import Encoder
from modules.dataset import tiny_words

parser = argparse.ArgumentParser(
    description="Train an Tacotron model for speech synthesis")
parser.add_argument("--max-epochs", type=int, default=100000)
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true')
parser.set_defaults(use_cuda=False)


def train_single_batch(input_variable, target_variable, 
    encoder, decoder,
    encoder_optimizer, decoder_optimizer, criterion,
    teacher_forcing_ratio = 0.5,
    clip = 5.0):
    """
    Args:
        input_variable: A Tensor of size (batch_size, max_text_length)
        target_variable: A Tensor of size 
            (batch_size, max_audio_length, frame_size)
    """

    # zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # added onto for each frame 
    loss = 0 

    # get batch size and initialize GO_frame
    batch_size = input_variable.size()[0]
    GO_frame = np.zeros((batch_size, decoder.frame_size))

    # get target length 
    target_length = target_variable.size()[1]

    # Run words through encoder
    encoder_outputs = encoder(input_variable)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.from_numpy(GO_frame).float())
    attn_gru_hidden, decoder_gru_hiddens = decoder.init_hiddens(batch_size)  

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        
        # Teacher forcing: Use the ground-truth target as the next input
        for t in range(int(target_length / decoder.num_frames)):
            decoder_output, attn_gru_hidden, decoder_gru_hiddens, attn_weights = \
                decoder(decoder_input, attn_gru_hidden, decoder_gru_hiddens, encoder_outputs)
            predict_frames = decoder_output.view(
                batch_size, decoder.num_frames, decoder.frame_size).clone()
            truth_frames = \
                target_variable[:, decoder.num_frames * t:decoder.num_frames * (t+1), :].clone()
            loss += criterion(predict_frames, truth_frames) 
            # use truth
            decoder_input = target_variable[:, decoder.num_frames * (t+1) - 1, :].contiguous().clone()

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for t in range(int(target_length / decoder.num_frames)):
            decoder_output, attn_gru_hidden, decoder_gru_hiddens, attn_weights = \
                decoder(decoder_input, attn_gru_hidden, decoder_gru_hiddens, encoder_outputs)
            predict_frames = decoder_output.view(
                batch_size, decoder.num_frames, decoder.frame_size).clone()
            truth_frames = \
                target_variable[:, decoder.num_frames * t:decoder.num_frames * (t+1), :].clone()
            loss += criterion(predict_frames, truth_frames) 
            # use predict 
            decoder_input = predict_frames[:, -1, :].contiguous().clone()

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

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
    start_loading_time = time.time()
    print('loading dataset...')
    ds = tiny_words()
    print('took %.3fs' % (time.time() - start_loading_time))

    # initialize model
    embedding_dim = 256
    bank_k = 16
    bank_ck = 128
    proj_dims = (128, 128)
    highway_layers = 4
    highway_units = 128
    gru_units = 128
    encoder = Encoder(ds.lang.num_chars, embedding_dim,
        bank_k, bank_ck, proj_dims, highway_layers,
        highway_units, gru_units)

    decoder = AttnDecoder(ds.max_text_length, use_cuda=args.use_cuda)

    if args.use_cuda:
        encoder.cuda()
        decoder.cuda()

    # initialize optimizers and criterion
    learning_rate = 0.0001
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    # configuring traingin
    n_epochs = args.max_epochs
    plot_every = 200
    print_every = 100
    batch_size = 32

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every

    for epoch in range(1, n_epochs + 1):
        
        # get training data for this cycle
        spectros, indexed_texts = ds.next_batch(batch_size)
        input_variable = Variable(torch.from_numpy(indexed_texts))
        target_variable = Variable(torch.from_numpy(spectros).float())

        if args.use_cuda:
            input_variable.cuda()
            target_variable.cuda()

        # train single batch 
        loss = train_single_batch(input_variable, 
            target_variable, encoder, decoder, 
            encoder_optimizer, decoder_optimizer, 
            criterion)

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

def main():
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
