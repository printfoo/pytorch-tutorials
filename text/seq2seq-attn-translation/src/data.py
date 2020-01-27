import torch
import glob
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

# Find all files given a pattern.
def findFiles(path):
    return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split("\n")
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
def loadData(path="data/names/*.txt"):
    category_lines = {}
    all_categories = []
    for filename in findFiles(path):
        category = filename.split("/")[-1].split(".")[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    n_categories = len(all_categories)
    return category_lines, all_categories

# One-hot vector for category
def categoryTensor(category, all_categories):
    li = all_categories.index(category)
    tensor = torch.zeros(1, len(all_categories))
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):

    # If not found, index = -1
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

if __name__ == "__main__":
    eg = unicodeToAscii("O'Néàl")
    print("Name:", eg)
    print("Name Tensor:", inputTensor(eg))
    print("Char ID:", targetTensor(eg))
    eg_cat = "Irish"
    category_lines, all_categories = loadData()
    n_categories = len(all_categories)
    print("Category:", eg_cat)
    print("Category Tensor:", categoryTensor(eg_cat, all_categories))
