import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from modules.commons import _wx
from modules.prenet import PreNet

class AttnDecoder(nn.Module):

    def __init__(self, max_text_length,
        attn_gru_hidden_size=256, 
        frame_size=80, num_frames=3,
        decoder_gru_hidden_size=256,
        decoder_num_layers=2,
        use_cuda=False):
        super(AttnDecoder, self).__init__()
        self.attn_gru_hidden_size = attn_gru_hidden_size
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.decoder_gru_hidden_size = decoder_gru_hidden_size
        self.decoder_output_size = frame_size * num_frames 
        self.decoder_num_layers = decoder_num_layers 
        self.max_text_length = max_text_length
        self.use_cuda = use_cuda

        # initialize layers
        # prenet
        self.prenet = PreNet(frame_size)
        self.attn_gru_input_size = self.prenet.fc2_hidden_size

        # attn gru 
        self.attn_gru = nn.GRUCell(self.attn_gru_input_size,
                                   self.attn_gru_hidden_size)
        # decoder gru
        self.decoder_grus = []
        for i in range(self.decoder_num_layers):
            self.decoder_grus.append(
                nn.GRUCell(self.decoder_gru_hidden_size,
                           self.decoder_gru_hidden_size))

        # initialize weights used in attention 
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
        self.w1 = nn.Parameter(torch.Tensor(self.attn_gru_hidden_size,
                                        self.attn_gru_hidden_size))
        init.normal(self.w1)

        self.w2 = nn.Parameter(torch.Tensor(self.attn_gru_hidden_size,
                                        self.attn_gru_hidden_size))
        init.normal(self.w2)

        self.v = nn.Parameter(torch.Tensor(self.attn_gru_hidden_size))
        init.normal(self.v)

        # other layers
        self.attn_combine = nn.Linear(self.attn_gru_hidden_size * 2, self.attn_gru_hidden_size)

        self.out = nn.Linear(self.decoder_gru_hidden_size, 
                             self.decoder_output_size)

    def forward(self, input, attn_gru_hidden, decoder_gru_hiddens, encoder_outputs):
        """
        Args:
            input: last output frame from previous time step 
                with size (batch_size, frame_size) 
            attn_gru_hidden: hidden state of attn rnn,
                with size (batch_size, attn_gru_hidden_size) 
            decoder_gru_hiddens: A list of Tensors of length decoder_num_layers,
                each with size (batch_size, decoder_gru_hidden_size)
            encoder_outputs: A Tensor of size (batch_size,
                                               time_steps, 
                                               2 * encoder_hidden_size)
                time_steps = self.max_text_length
                2 * encoder_hidden_size = attn_gru_hidden_size

        Returns:
            output: A Tensor of size (batch_size, decoder_output_size)
            attn_gru_hidden: See above
            decoder_gru_hiddens: See above
            a: Attention weights, a Tensor of size (batch_size, max_text_length)
        """
        batch_size = input.size()[0] 
        pre_out  = self.prenet(
            input.view(batch_size, 1, self.frame_size)
        ).squeeze()

        # run attn gru for one step
        # attn_output has size (batch_size, self.attn_gru_hidden_size)
        new_attn_gru_hidden = self.attn_gru(pre_out, attn_gru_hidden)
        attn_output = F.relu(new_attn_gru_hidden)

        # please refer to the following paper for attention equations
        # https://papers.nips.cc/paper/5635-grammar-as-a-foreign-language.pdf

        # dt has size (batch_size, self.attn_gru_hidden_size, self.max_text_length)
        dt = attn_output.unsqueeze(2).expand(
            batch_size, self.attn_gru_hidden_size, self.max_text_length)

        # v has size (batch_size, 1, self.attn_gru_hidden_size)
        v = self.v.unsqueeze(0).unsqueeze(1).expand(
            batch_size, 1, self.attn_gru_hidden_size)

        # u has size (batch_size, 1, self.max_text_length)
        u = v.bmm(F.tanh(_wx(self.w1, encoder_outputs.transpose(1, 2)) +
                         _wx(self.w2, dt)))

        # a is attention weight vector, (batch_size, self.max_text_length)
        a = F.softmax(u.squeeze(1))

        # dtp has size (batch_size, 1, self.attn_gru_hidden_size)
        dtp = torch.sum(
            encoder_outputs * a.unsqueeze(2).expand(
                batch_size, self.max_text_length, self.attn_gru_hidden_size), 1)

        # decoder_input has size (batch_size, self.attn_gru_hidden_size)
        decoder_input = self.attn_combine(
            torch.cat((attn_output, dtp.squeeze()), 1))
        decoder_output = decoder_input

        new_decoder_gru_hiddens = []
        for i in range(self.decoder_num_layers):
            decoder_hidden = self.decoder_grus[i](decoder_output, 
                                                  decoder_gru_hiddens[i])
            new_decoder_gru_hiddens.append(decoder_hidden)
            decoder_output = decoder_output + F.relu(decoder_hidden)

        output = F.softmax(self.out(decoder_output))
        return output, new_attn_gru_hidden, new_decoder_gru_hiddens, a

    def init_hiddens(self, batch_size):
        attn_gru_hidden = Variable(torch.zeros(batch_size, 
                                           self.attn_gru_hidden_size))
        decoder_gru_hiddens = []
        for i in range(self.decoder_num_layers):
            decoder_gru_hiddens.append(
                Variable(torch.zeros(batch_size, self.decoder_gru_hidden_size)))
        if self.use_cuda:
            attn_gru_hidden.cuda()
            map(lambda x: x.cuda(), decoder_gru_hiddens)
        return attn_gru_hidden, decoder_gru_hiddens

