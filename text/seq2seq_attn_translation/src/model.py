from data import *

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# RNN encoder.
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
    
        # embedded - (batch_size, seq_len, idden_size).
        embedded = self.embedding(input).view(1, 1, -1)
 
        # output - (batch_size, seq_len, hidden_size).
        # hidden - (batch_size, seq_len, hidden_size).
        output, hidden = self.rnn(embedded, hidden)
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# RNN decoder.
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = F.relu(embedded)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# RNN decoder with attention.
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
    
        # embedded - (batch_size, seq_len, hidden_size).
        embedded = self.embedding(input).view(1, 1, -1)
        
        # attn_weights - (batch_size, 1, max_len).
        attn_weights = self.attn(torch.cat((embedded, hidden), 2))
        attn_weights = F.softmax(attn_weights, dim=2)
        
        # encoder_outputs - (batch_size, max_len, hidden_size).
        encoder_outputs = encoder_outputs.unsqueeze(0)
        
        # attn_applied - (batch_size, 1, hidden_size).
        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        # output - (batch_size, 1, hidden_size).
        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
