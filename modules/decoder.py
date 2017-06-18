import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from modules.commons import _wx

class AttnDecoder(nn.Module):

    def __init__(self, attn_input_size=128, attn_hidden_size=256,
        decoder_output_size=240, decoder_hidden_size=256, 
        decoder_num_layers=2, max_length=30, use_cuda=False):
        super(AttnDecoder, self).__init__()
        self.attn_input_size = attn_input_size
        self.attn_hidden_size = attn_hidden_size

        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_output_size = decoder_output_size
        self.decoder_num_layers = decoder_num_layers 

        self.max_length = max_length
        self.use_cuda = use_cuda

        # initialize gru cells
        self.attn_gru = nn.GRUCell(self.attn_input_size,
                                   self.attn_hidden_size)
        self.decoder_grus = []
        for i in range(self.decoder_num_layers):
            self.decoder_grus.append(
                nn.GRUCell(self.decoder_hidden_size,
                           self.decoder_hidden_size))

        # initialize weights
        self.w1 = Variable(torch.Tensor(self.attn_hidden_size,
                                        self.attn_hidden_size))
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
        init.normal(self.w1)
        self.w2 = Variable(torch.Tensor(self.attn_hidden_size,
                                        self.attn_hidden_size))
        init.normal(self.w2)
        self.v = Variable(torch.Tensor(self.attn_hidden_size))
        init.normal(self.v)

        # initialize other layers
        self.attn_combine = nn.Linear(self.attn_hidden_size * 2, self.attn_hidden_size)

        self.out = nn.Linear(self.decoder_hidden_size, 
                             self.decoder_output_size)

    def forward(self, input, attn_hidden, decoder_hiddens, encoder_outputs):
        """
        Args:
            input: output at time step t from pre-net,
                with size (batch_size, attn_input_size)
            attn_hidden: hidden state of attn rnn,
                with size (batch_size, attn_hidden_size) 
            decoder_hiddens: A list of Tensors of length decoder_num_layers,
                each with size (batch_size, decoder_hidden_size)
            encoder_outputs: A Tensor of size (batch_size,
                                               time_steps, 
                                               2 * encoder_hidden_size)
                time_steps = self.max_length
                2 * encoder_hidden_size = attn_hidden_size

        Returns:
            output: A Tensor of size (batch_size, decoder_output_size)
            attn_hidden: See above
            decoder_hiddens: See above
            a: Attention weights, a Tensor of size (batch_size, max_length)
        """
        batch_size = input.size()[0] 

        # run attn gru for one step
        # attn_output has size (batch_size, self.attn_hidden_size)
        attn_hidden = self.attn_gru(input, attn_hidden)
        attn_output = F.relu(attn_hidden)

        # please refer to the following paper for attention equations
        # https://papers.nips.cc/paper/5635-grammar-as-a-foreign-language.pdf

        # dt has size (batch_size, self.attn_hidden_size, self.max_length)
        dt = attn_output.unsqueeze(2).expand(
            batch_size, self.attn_hidden_size, self.max_length) 

        # u has size (batch_size, 1, self.attn_hidden_size)
        v = self.v.unsqueeze(0).unsqueeze(1).expand(
            batch_size, 1, self.attn_hidden_size)

        # u has size (batch_size, 1, self.max_length)
        u = v.bmm(F.tanh(_wx(self.w1, encoder_outputs.transpose(1, 2)) +
                         _wx(self.w2, dt))) 

        # a is attention weight vector, (batch_size, self.max_length)
        a = F.softmax(u.squeeze(1))

        # dtp has size (batch_size, 1, self.attn_hidden_size)
        dtp = torch.sum(encoder_outputs * a.unsqueeze(2).expand(
            batch_size, self.max_length, self.attn_hidden_size), 1)

        # decoder_input has size (batch_size, self.attn_hidden_size)
        decoder_input = self.attn_combine(
            torch.cat((attn_output, dtp.squeeze()), 1))
        decoder_output = decoder_input

        for i in range(self.decoder_num_layers):
            decoder_hidden = self.decoder_grus[i](decoder_output, 
                                                  decoder_hiddens[i])
            decoder_hiddens[i] = decoder_hidden
            decoder_output += F.relu(decoder_hidden)

        output = F.softmax(self.out(decoder_output))
        return output, attn_hidden, decoder_hiddens, a

    def init_hiddens(self, batch_size):
        attn_hidden = Variable(torch.zeros(batch_size, 
                                           self.attn_hidden_size))
        decoder_hiddens = []
        for i in range(self.decoder_num_layers):
            decoder_hiddens.append(
                Variable(torch.zeros(batch_size, self.decoder_hidden_size)))
        if self.use_cuda:
            attn_hidden.cuda()
            map(lambda x: x.cuda(), decoder_hiddens)
        return attn_hidden, decoder_hiddens

