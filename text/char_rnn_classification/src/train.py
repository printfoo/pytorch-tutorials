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

# Get the category and index of the greatest value.
def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

# Make a random choice in a list.
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Randomly select an example to train.
def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()  # Initialize hidden layer.
    rnn.zero_grad()  # Initialize RNN with zero gradient.

    for i in range(line_tensor.size()[0]):  # One pass of the input sequence.
        output, hidden = rnn(line_tensor[i], hidden)

    # Computes gradients of the loss wrt parameters using backpropagation.
    loss = criterion(output, category_tensor)
    loss.backward()

    # Update parameters based on gradients and learning rate.
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

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
    rnn = RNN(n_letters, n_hidden, n_categories)  # Initialize RNN.

    # Define loss and learning rate.
    criterion = nn.NLLLoss()  # Negative log-likelihood loss.
    learning_rate = 0.005

    # Start training.
    start_time = time.time()  # Start timing.
    n_iters = 100000
    print_every = 1000
    for iter in range(1, n_iters + 1):  # Train n_iters iterations.
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)

        # Print iter number, loss, name and predict.
        if iter % print_every == 0:
            predict, predict_i = categoryFromOutput(output)
            result = "✓" if predict == category else "✗ (%s)" % category
            print("%d %d%% (%s) %.4f %s / %s %s" % (iter, iter / n_iters * 100, timeSince(start_time), loss, line, predict, result))

    # Save the trained model.
    torch.save(rnn, args.output)
