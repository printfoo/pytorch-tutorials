from model import *
from data import *
import sys

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    # Pass the RNN for the input line.
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(line, n_predictions=3):
    output = evaluate(Variable(lineToTensor(line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print("(%.2f) %s" % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions

if __name__ == "__main__":
    rnn = torch.load("char-rnn-classification.pt")
    category_lines, all_categories = loadData("../data/names/*.txt")
    predict(sys.argv[1])
