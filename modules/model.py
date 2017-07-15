import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.encoder import Encoder
from modules.decoder import AttnDecoder
from modules.cbhg import CBHG


class Tacotron(nn.Module):

    def __init__(self, num_embeddings, hp):
        self.hp = hp
        self.encoder = Encoder(
            num_embeddings, hp.embedding_dim, hp.encoder_bank_k,
            hp.encoder_bank_ck, hp.encoder_proj_dims,
            hp.encoder_highway_layers, hp.encoder_gru_units,
            dropout=hp.droput, use_cuda=hp.use_cuda
        )

        self.decoder = AttnDecoder(
            hp.max_text_length, hp.attn_gru_hidden_size, hp.n_mels,
            hp.reduction_factor, hp.decoder_gru_hidden_size,
            hp.decoder_gru_layers, dropout=hp.dropout, use_cuda=hp.use_cuda
        )

        self.postprocess = CBHG(
            hp.n_mels, hp.post_bank_k, hp.post_bank_ck,
            hp.post_proj_dims, hp.post_highway_layers, hp.post_highway_units,
            hp.post_gru_units, use_cuda=hp.use_cuda
        ) 

    def forward(self, x):
        """
        Args:
            x: A Tensor of size (batch_size, time_steps)

        Returns:
            Predicted spectrogram of size (batch_size, )
        """
        encoder_outputs = self.encoder(x)

        GO_frame = np.zeros((batch_size, self.decoder.frame_size))
        decoder_input = Variable(torch.from_numpy(GO_frame).float())
        if self.hp.use_cuda:
            decoder_input = decoder_input.cuda()
        attn_gru_hidden, decoder_gru_hiddens = \
            self.decoder.init_hiddens(self.hp.batch_size)

        use_teacher_forcing = random.random() < self.hp.teacher_forcing_ratio
        if use_teacher_forcing:

            for t in range(int(target_length))


