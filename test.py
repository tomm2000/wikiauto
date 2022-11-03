# future
from __future__ import unicode_literals, print_function, division
from turtle import st
from lib.beam_search import beam_search

# other files
from lib.vocab import END_TOKEN, PADDING_TOKEN, START_TOKEN
from lib.generic import *
from lib.load_data import batchPair, getInputSizeAverage, load_data, testBatch
from lib.models import EncoderRNN, AttnDecoderRNN
from lib.load_setup import *

# pytorch
import torch
import torch.nn as nn
from torch import le, optim
from ignite.metrics.nlp import Bleu, Rouge

# other
import random
import time
from colorama import Fore
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ----============= SETUP =============----
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # DEBUG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # DEBUG

print(Fore.MAGENTA + f"Using device:{Fore.RESET} '{device}'")

SETUP_FOLDER = "setup/maintest"
SETUP, DATA = load_setup(SETUP_FOLDER)

print(Fore.MAGENTA + f"Loaded data from previous training session:{Fore.RESET} {SETUP['epoch']} epochs trained")

BATCH_SIZE = SETUP["batch_size"]
ENCODER_INPUT_SIZE = SETUP["encoder_input_size"]
HIDDEN_SIZE = SETUP["hidden_size"]
EMBEDDING_SIZE = SETUP["embedding_size"]
VOCAB_SIZE = SETUP["vocab_size"]
DECODER_OUTPUT_SIZE = SETUP["decoder_output_size"]
PAIR_AMOUNT = SETUP["pairs"]
TEACHER_FORCIING_RATIO = SETUP["teacher_forcing_ratio"]

# ----============= DATA LOADING =============----
print(Fore.MAGENTA + "\n---- Loading data ----" + Fore.RESET)

type_vocab, value_vocab, token_vocab, train_split, eval_split, test_split = load_data(
  device=device,
  vocab_size=VOCAB_SIZE,
  input_size=ENCODER_INPUT_SIZE,
  output_size=DECODER_OUTPUT_SIZE,
  pair_amount=PAIR_AMOUNT,
)

# ----============= TEST FUNCTIONS =============----
def testEpoch(encoder, decoder, inputs):
  encoder.eval()
  decoder.eval()

  test_amount = len(inputs[0])

  bleu = Bleu(ngram=4, smooth="smooth1")
  rouge = Rouge()

  for i in tqdm(range(test_amount), desc="Testing: "):
    input_tensor, target_tensor, attn_mask = testBatch(inputs, i, BATCH_SIZE)

    encoder_hidden = encoder.initHidden(device, BATCH_SIZE)

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

    out = beam_search(
      decoder=decoder,
      batch_size=BATCH_SIZE,
      beam_size=5,
      seq_len=DECODER_OUTPUT_SIZE,
      encoder_outputs=encoder_outputs,
      attn_mask=attn_mask,
      encoder_hidden=encoder_hidden,
      encoder_input_size=ENCODER_INPUT_SIZE,
      end_id=token_vocab.getID(END_TOKEN),
      start_id=token_vocab.getID(START_TOKEN),
      device=device,
    )

    out = [token_vocab.getWord(i.item()) for i in out]
    target = [token_vocab.getWord(i.item()) for i in target_tensor[0]]

    bleu.update((out, target))
    rouge.update((out, target))

  return bleu.compute(), rouge.compute()
    
# ----============= MODEL LOADING =============----
print(Fore.MAGENTA + "\n---- Loading model ----" + Fore.RESET)

START_EPOCH =  SETUP["epoch"]
model_path = SETUP["model_path"]
result_path = SETUP["result_path"]

encoder = f"{model_path}/encoder_{START_EPOCH}.pt"
decoder = f"{model_path}/decoder_{START_EPOCH}.pt"

encoder = torch.load(encoder)
decoder = torch.load(decoder)

encoder.to(device)
decoder.to(device)

score = testEpoch(encoder, decoder, test_split)

print(Fore.MAGENTA + f"BLEU:{Fore.RESET} {score[0]}")
print(Fore.MAGENTA + f"ROUGE:{Fore.RESET} {score[1]}")
