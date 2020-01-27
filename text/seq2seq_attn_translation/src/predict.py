import time
import math
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import torch
from torch import optim

from data import *
from model import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=False)
parser.add_argument("--figs_dir", type=str, required=False)
args, extras = parser.parse_known_args()
args.extras = extras
args.command = " ".join(["python"] + sys.argv)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def showAttention(input_sentence, output_words, attentions):

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(os.path.join(args.figs_dir, " ".join(output_words) + ".pdf"))

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print("Input\t", pair[0])
        print("Truth\t", pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = " ".join(output_words[:-1])
        print("Predict\t", output_sentence)
        print()

        # Visualize attention.
        showAttention(pair[0], output_words, attentions)

if __name__ == "__main__":

    # Load data.
    input_lang, output_lang, pairs = prepareData("eng", "fra", True)

    # Load saved model.
    encoder = torch.load(os.path.join(args.output_dir, "encoder.pt"))
    attn_decoder = torch.load(os.path.join(args.output_dir, "decoder.pt"))

    # Evaluate some.
    evaluateRandomly(encoder, attn_decoder)
