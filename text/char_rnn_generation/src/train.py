import torch
from data import *
from model import *
import random
import time
import math
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=False)
parser.add_argument("--output", type=str, required=False)
args, extras = parser.parse_known_args()
args.extras = extras
args.command = " ".join(["python"] + sys.argv)

# Make a random choice in a list.
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category, all_categories)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)  # Transpose.
    hidden = rnn.initHidden()  # Initialize hidden layer.
    rnn.zero_grad()  # Initialize RNN with zero gradient.
    loss = 0  # Initialize loss to 0.

    for i in range(input_line_tensor.size(0)):  # One pass of the input sequence.
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l  # Accumulate loss of the input sequence.

    # Computes gradients of the loss wrt parameters using backpropagation.
    loss.backward()

    # Update parameters based on gradients and learning rate.
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

# Time since training.
def timeSince(start_time):
    s = time.time() - start_time
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

if __name__ == "__main__":

    # Load data and initialize RNN.
    category_lines, all_categories = loadData(args.data)
    n_categories = len(all_categories)
    n_letters = n_letters
    n_hidden = 128
    rnn = RNN(n_categories, n_letters, n_hidden, n_letters)  # Initialize RNN.

    # Define loss and learning rate.
    criterion = nn.NLLLoss()  # Negative log-likelihood loss.
    learning_rate = 0.0005

    # Start training.
    start_time = time.time()  # Start timing.
    n_iters = 10000
    print_every = 1000
    for iter in range(1, n_iters + 1):  # Train n_iters iterations.
        output, loss = train(*randomTrainingExample())

        # Print iter number, loss, name and predict.
        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start_time), iter, iter / n_iters * 100, loss))

    # Save the trained model.
    torch.save(rnn, args.output)
