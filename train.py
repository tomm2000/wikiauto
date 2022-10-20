import torch
from torch import optim
import torch.nn as nn

import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from lib.generic import plotResults
from lib.load_data import load_data, load_vocab, make_batch
from lib.models import DecoderRNN, EncoderRNN, SeqToSeq
from lib.persistance import load_results, load_setup, save_results, save_sample
from lib.vocab import PADDING_TOKEN

# --------------------------- SETUP ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: '{device}'")

SETUP = load_setup("setup/test4.json")

# --------------------------- DATA LOADING ---------------------------
print("--------------------------- DATA LOADING ---------------------------")

type_vocab, value_vocab, token_vocab = load_vocab(
  folder=SETUP["data_folder"] + "counts/",
  vocab_size=SETUP["vocab_size"]
)

train_data = load_data(
  dataset=SETUP["data_folder"] + "dataset/train.json",
  device=device,
  input_size=SETUP["input_size"],
  output_size=SETUP["output_size"],
  articles=SETUP["articles"],
  type_vocab=type_vocab,
  value_vocab=value_vocab,
  token_vocab=token_vocab
)

valid_data = load_data(
  dataset=SETUP["data_folder"] + "dataset/validation.json",
  device=device,
  input_size=SETUP["input_size"],
  output_size=SETUP["output_size"],
  articles=-1,
  type_vocab=type_vocab,
  value_vocab=value_vocab,
  token_vocab=token_vocab
)

test_data = load_data(
  dataset=SETUP["data_folder"] + "dataset/test.json",
  device=device,
  input_size=SETUP["input_size"],
  output_size=SETUP["output_size"],
  articles=-1,
  type_vocab=type_vocab,
  value_vocab=value_vocab,
  token_vocab=token_vocab
)

# --------------------------- MODEL LOADING ---------------------------
print("--------------------------- MODEL LOADING ---------------------------")

RESULTS = load_results("results/main.json")

encoder = f"{SETUP['model_folder']}encoder_{RESULTS['epoch']}.pt"
decoder = f"{SETUP['model_folder']}decoder_{RESULTS['epoch']}.pt"

print("looking for encoder in: " + str(encoder))
print("lookind for decoder in: " + str(decoder))

try:
  encoder = torch.load(encoder)
  print("loaded old encoder")
except:
  print("could not load old encoder, creating new one")
  encoder = EncoderRNN(
    type_vocab=type_vocab,
    value_vocab=value_vocab,
    hidden_size=SETUP['hidden_size'],
    embedding_size=SETUP['embedding_size'],
    input_size=SETUP['input_size'],
  ).to(device)

try:
  decoder = torch.load(decoder)
  print("loaded old decoder")
except:
  print("could not load old decoder, creating new one")
  decoder = DecoderRNN(
    output_vocab_size=len(token_vocab),
    hidden_size=SETUP['hidden_size'],
    embedding_size=SETUP['embedding_size'],
    batch_size=SETUP['batch_size'],
    input_size=SETUP['input_size'],
    device=device,
  ).to(device)
  
seq2seq = SeqToSeq(
  encoder=encoder,
  decoder=decoder,
  batch_size=SETUP['batch_size'],
  device=device,
  token_vocab=token_vocab,
  input_size=SETUP['input_size'],
  output_size=SETUP['output_size'],
  teacher_forcing_ratio=SETUP['teacher_forcing_ratio'],
)

# --------------------------- TRAIN/VALIDATION/TEST FUNCTIONS ---------------------------
def run_epoch(model, data, mode = 'valid', learning_rate = None):
  if mode == 'train':
    model.set_train()
    desc = "Training: "
    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=learning_rate)
  if mode == 'valid':
    desc = "Validating: "
    model.set_eval()

  epoch_len = math.floor(len(data) / SETUP['batch_size'])
  
  if epoch_len == 0:
    raise Exception(f"epoch len == 0, inputs size [{data[0].size(0)}] / batch size [{SETUP['batch_size']}] must be >= 1")

  criterion = nn.NLLLoss(reduction='none', ignore_index=token_vocab.getID(PADDING_TOKEN))

  tot_loss = 0.0
  tot_perplexity = 0.0
  sample = None

  for i in tqdm(range(0, epoch_len), desc=desc):
    inputs, target, attn_mask = make_batch(data, i, SETUP['batch_size'], type_vocab.getID(PADDING_TOKEN))

    if mode == 'train':
      output, loss = model.train(
        encoder_input=inputs,
        target=target,
        attn_mask=attn_mask,
        criterion=criterion,
        encoder_optimizer=encoder_optimizer,
        decoder_optimizer=decoder_optimizer
      )
    if mode == 'valid':
      output, loss = model.predict(
        encoder_input=inputs,
        attn_mask=attn_mask,
        criterion=criterion,
        target=target,
      )

    tot_loss += loss.item()
    tot_perplexity += torch.exp(loss).item()

    if i == 0: sample = (output, target)
  
  epoch_loss = tot_loss / epoch_len
  epoch_perplexity = tot_perplexity / epoch_len

  return sample, epoch_loss, epoch_perplexity
  

# --------------------------- TRAINING LOOP ---------------------------

while True:
  print(f"--------------------------- EPOCH {RESULTS['epoch']} ---------------------------")

  random.shuffle(train_data)

  # train epoch
  train_sample, train_loss, train_perplexity = run_epoch(seq2seq, train_data, 'train', SETUP['learning_rate'])
  RESULTS['train_losses'].append(train_loss)
  RESULTS['train_perplexities'].append(train_perplexity)

  # validate epoch
  valid_sample, valid_loss, valid_perplexity = run_epoch(seq2seq, valid_data, 'valid')
  RESULTS['valid_losses'].append(valid_loss)
  RESULTS['valid_perplexities'].append(valid_perplexity)

  print("--------------------------- SAVING RESULTS ---------------------------")
  print(f"Train loss: {train_loss}, Train perplexity: {train_perplexity}")
  print(f"Valid loss: {valid_loss}, Valid perplexity: {valid_perplexity}")

  # save model
  torch.save(seq2seq.encoder, f"{SETUP['model_folder']}encoder_{RESULTS['epoch']}.pt")
  torch.save(seq2seq.decoder, f"{SETUP['model_folder']}decoder_{RESULTS['epoch']}.pt")

  # save losses
  save_results(
    file=f"{SETUP['results_folder']}data.json",
    data=RESULTS
  )

  # save sample
  save_sample(
    file=f"{SETUP['results_folder']}outputs.txt",
    sample=train_sample,
    epoch=RESULTS['epoch'],
    vocab=token_vocab
  )

  # plot
  plotResults(
    train_losses=RESULTS['train_losses'],
    train_perplexities=RESULTS['train_perplexities'],
    valid_losses=RESULTS['valid_losses'],
    valid_perplexities=RESULTS['valid_perplexities'],
  ).savefig(f"{SETUP['results_folder']}plot.png")
  plt.close('all')

  RESULTS['epoch'] += 1

  






