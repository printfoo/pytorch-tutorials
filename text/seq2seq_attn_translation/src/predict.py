from model import *
from data import *
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=False)
parser.add_argument("--output", type=str, required=False)
args, extras = parser.parse_known_args()
args.extras = extras
args.command = " ".join(["python"] + sys.argv)

# Sample from a category and starting letter
def sample(category, start_letter="A", max_length=50):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category, all_categories)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:  # EOS.
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)  # Next.

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters="A"):
    for start_letter in start_letters:
        output_name = sample(category, start_letter)
        print(output_name, end = " ")

if __name__ == "__main__":
    category_lines, all_categories = loadData(args.data)
    rnn = torch.load(args.output)
    for category in all_categories:
        print("\n" + category)
        samples(category, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    print()
