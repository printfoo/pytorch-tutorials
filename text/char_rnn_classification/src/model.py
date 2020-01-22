from data import *
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # nn.Linear(in_features, out_features, bias=True) -- 
        # linear transformation y = ax + b with initialized weight to learn.
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Input to hidden.
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # Input to output.

        # nn.Softmax(dim=None) -- rescaling.
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        # Concatenates tensors in 1st dimension, size = input + hidden.
        combined = torch.cat(tensors=(input, hidden), dim=1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

if __name__ == "__main__":

    # Initialize RNN.
    n_hidden = 128
    n_categories = loadData("../data/names/*.txt")
    print("N letters:", n_letters)  # 58
    print("N hidden:", n_hidden)  # 128
    print("N categories:", n_categories)  # 18
    rnn = RNN(n_letters, n_hidden, n_categories)
    print("RNN:", rnn)

    # Forward propagation.
    input = lineToTensor("Shan")  # 4 * 1 * 58
    hidden = torch.zeros(1, n_hidden)  # 1 * 128
    print("First input:", input[0])  # 1 * 58
    print("Hidden:", hidden)  # 1 * 128
    output, next_hidden = rnn(input[0], hidden)
    print("Output:", output)  # 1 * 18
    print("Next hidden:", next_hidden)  # 1 * 128
