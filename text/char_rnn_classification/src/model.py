import data
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # input to hidden.
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # input to output.
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

if __name__ == "__main__":
    n_letters = data.n_letters
    n_hidden = 128
    n_categories = data.loadData("../data/names/*.txt")
    print(n_letters, n_hidden, n_categories)
    rnn = RNN(n_letters, n_hidden, n_categories)
    print(rnn)

    input = data.lineToTensor("Shan")
    hidden = torch.zeros(1, n_hidden)
    output, next_hidden = rnn(input[0], hidden)
    print(output)
