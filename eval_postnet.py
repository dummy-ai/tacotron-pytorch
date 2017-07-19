import torch
import torch.nn as nn
import argparse
import sys
import numpy as np
from torch.autograd import Variable
from modules.decoder import AttnDecoder
from modules.encoder import Encoder
from modules.postnet import PostNet
from modules.dataset import tiny_words, indexes_from_text, pad_indexes
from modules.audio_signal import spectrogram2wav, griffinlim
from modules.hyperparams import Hyperparams as hp
from scipy.io.wavfile import write

EOT_token = 0
PAD_token = 1

parser = argparse.ArgumentParser(
    description="Generate wav based on melspectrogram")
parser.add_argument('-d', '--data-size', default=sys.maxsize, type=int)
parser.add_argument("--checkpoint", type=str, default="tacotron.checkpoint")
args = parser.parse_args()

hp.use_cuda = False

def inference(checkpoint_file):
    ds = tiny_words(
        max_text_length=hp.max_text_length,
        max_audio_length=hp.max_audio_length,
        max_dataset_size=args.data_size
    )

    postnet = PostNet(
        hp.n_mels, 1 + hp.n_fft//2,
        hp.post_bank_k, hp.post_bank_ck,
        hp.post_proj_dims, hp.post_highway_layers, hp.post_highway_units,
        hp.post_gru_units, use_cuda=hp.use_cuda
    )

    postnet.eval()

    if hp.use_cuda:
        postnet.cuda()

    # load model
    checkpoint = torch.load(checkpoint_file)
    postnet.load_state_dict(checkpoint['postnet'])

    mels, mags, indexed_texts = ds.next_batch(1)
    mels_v = Variable(torch.from_numpy(np.repeat(mels, 1, 0)).float())
    mags_v = Variable(torch.from_numpy(np.repeat(mags, 1,0)).float())
    if hp.use_cuda:
        mels_v = mels_v.cuda()
        mags_v = mags_v.cuda()

    # postnet
    post_out = postnet(mels_v)
    s = post_out[0].cpu().data.numpy()

    criterion = nn.L1Loss()
    print("output", post_out)
    print("truth", mags)
    print("Loss = ", criterion(post_out, mags_v).data[0] / hp.max_audio_length)

    print("Recontructing wav...")
    #import pdb; pdb.set_trace();
    s = np.where(s < 0, 0, s)
    wav = spectrogram2wav(s**hp.power)
    # wav = griffinlim(s**hp.power)
    write("demo.wav", hp.sr, wav)


def main():
    inference(args.checkpoint)

if __name__ == "__main__":
    main()
