from data import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys

class RNN(nn.Module):
    def __init__(self, n_categories, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # nn.Linear(in_features, out_features, bias=True) -- 
        # linear transformation y = ax + b with initialized weight to learn.
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)  # Input to hidden.
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)  # Input to output.
        self.o2o = nn.Linear(hidden_size + output_size, output_size)  # Output to output.

        # nn.Dropout(prob) -- randomly zeros parts of input to prevent overfitting.
        self.dropout = nn.Dropout(0.1)

        # nn.Softmax(dim=None) -- rescaling.
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):

        # Encoder.
        input_combined = torch.cat(tensors=(category, input, hidden), dim=1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat(tensors=(hidden, output), dim=1)

        # Decoder.
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

if __name__ == "__main__":

    # Initialize RNN.
    n_hidden = 128
    category_lines, all_categories = loadData()
    n_categories = len(all_categories)
    print("N letters:", n_letters)  # 59
    print("N hidden:", n_hidden)  # 128
    print("N categories:", n_categories)  # 18
    rnn = RNN(n_categories, n_letters, n_hidden, n_letters)
    print("RNN:", rnn)

    # Forward propagation.
    category = categoryTensor("Irish", all_categories)  # 1 * 18
    input = inputTensor(unicodeToAscii("O'Néàl"))  # 6 * 1 * 59
    hidden = torch.zeros(1, n_hidden)  # 1 * 128
    
    output, next_hidden = rnn(category, input[0], hidden)
    print("Output:", output)  # 1 * 59
    print("Next hidden:", next_hidden)  # 1 * 128
