# future
from __future__ import unicode_literals, print_function, division

# other files
from helpers.vocab import START_TOKEN
from helpers.helpers import *
from helpers.load_data import batchPairs, load_data_training
from models import EncoderRNN, AttnDecoderRNN

# pytorch
import torch
import torch.nn as nn
from torch import optim

# other
import random
import time
from datetime import datetime
from colorama import Fore
import os
import matplotlib.pyplot as plt

# ----============= SETUP =============----
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # DEBUG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(Fore.MAGENTA + f"Using device:{Fore.RESET} '{device}'")

ENCODER_INPUT_SIZE = 40 # dimensione dell'input dell'encoder (numero di triple tipo-valore-posizione in input)
DECODER_OUTPUT_SIZE = 20 # dimensione dell'output del decoder (lunghezza della frase in output)
BATCH_SIZE = 5
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 128
PAIR_AMOUNT =  5
VOCAB_SIZE = 50000
teacher_forcing_ratio = 1 # 0 = no teacher forcing, 1 = only teacher forcing

data_path = "data"

# ----============= DATA LOADING =============----
print(Fore.MAGENTA + "\n---- Loading data ----" + Fore.RESET)

type_vocab, value_vocab, token_vocab, pairs = load_data_training(
  vocab_size=VOCAB_SIZE,
  input_size=ENCODER_INPUT_SIZE,
  output_size=DECODER_OUTPUT_SIZE,
  pair_amount=PAIR_AMOUNT,
  path=data_path
)

def split_data(pairs, train_size=0.8):
  train_size = int(train_size * len(pairs))
  train_pairs = pairs[:train_size]
  test_pairs = pairs[train_size:]
  return train_pairs, test_pairs

# ----============= TRAINING/EVALUATION FUNCTIONS =============----

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
  encoder_hidden = encoder.initHidden(device, BATCH_SIZE)

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  target_length = target_tensor.size(1)

  loss = 0

  # print(type_vocab.batchesToSentences(input_tensor[0], True))
  # print("1\n\n")
  # print(value_vocab.batchesToSentences(input_tensor[1], True))
  # print("2\n\n")
  # print(input_tensor[2])


  encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

  decoder_input = torch.tensor([type_vocab.getID(START_TOKEN) for _ in range(BATCH_SIZE)], device=device)
  decoder_hidden = encoder_hidden

  coverage = torch.zeros(BATCH_SIZE, ENCODER_INPUT_SIZE, device=device)
  context_vector = None

  # decoder_inputs = torch.zeros(BATCH_SIZE, target_length, device=device)

  for di in range(target_length):
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    current_target = target_tensor[:, di] # [BATCH]

    # decoder_inputs[:, di] = decoder_input

    if use_teacher_forcing: # Teacher forcing: Feed the target as the next input
      decoder_output, decoder_hidden, context_vector, attn_weights, coverage = decoder(encoder_outputs, decoder_input, decoder_hidden, coverage, context_vector)
      decoder_input = current_target
        
    else: # Without teacher forcing: use its own predictions as the next input
      decoder_output, decoder_hidden, context_vector, attn_weights, coverage = decoder(encoder_outputs, decoder_input, decoder_hidden, coverage, context_vector)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze()

    newloss = criterion(decoder_output, current_target)

    loss += newloss

  # print(token_vocab.batchesToSentences(decoder_inputs, True))
  # print("3\n\n")
  # print(token_vocab.batchesToSentences(target_tensor, True))
  # print("4\n\n")

  # raise Exception("STOP")

  loss = loss / target_length
  loss.backward()

  # Clip gradient
  # nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0, norm_type=2)
  encoder_optimizer.step()

  # nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
  decoder_optimizer.step()

  return loss.item()


def trainEpoch(encoder, decoder, inputs, print_times=10, plot_times=10000, learning_rate=5e-5):
  epoch_start = time.time()

  plot_losses = []
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every
  loss_total = 0

  epoch_len = len(inputs)
  plot_every = max(int(epoch_len / plot_times), 1)
  print_every = max(int(epoch_len / print_times), 1)

  encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
  criterion = nn.NLLLoss()

  avg_time = time.time()

  for iter in range(1, epoch_len+1):
    # ogni elemento di inputs è una tupla (input, target)
    # ogni valore input è un tensore di dimensione [3, batch, encoder_input_size], deve 3 rappresenta (tipo, valore, posizione)
    # ogni valore target è un tensore di dimensione [batch, decoder_output_size]

    training_pair = inputs[iter-1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    print_loss_total += loss
    plot_loss_total += loss
    loss_total += loss

    if iter % print_every == 0:
      print_loss_avg = print_loss_total / print_every

      average = asMsecs((time.time() - avg_time) / print_every)
      since = timeSince(epoch_start, iter / (epoch_len+1))
      info = f"({iter} / {iter / epoch_len * 100:.2f}%)"
      loss = f"{print_loss_avg:.4f}"

      print("Average time: " + average + " | " + since + " | " + info + " | Loss: " + loss)

      avg_time = time.time()
      print_loss_total = 0


    if iter % plot_every == 0:
      plot_loss_avg = plot_loss_total / plot_every
      plot_losses.append(plot_loss_avg)
      plot_loss_total = 0

  loss_avg = loss_total / epoch_len

  return plot_losses, loss_avg


def evaluate(input_tensor, target_tensor, encoder, decoder, criterion):

  encoder_hidden = encoder.initHidden(device, BATCH_SIZE)
  target_length = target_tensor.size(1)

  loss = 0

  encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

  decoder_input = torch.tensor([type_vocab.getID(START_TOKEN) for _ in range(BATCH_SIZE)], device=device)
  decoder_hidden = encoder_hidden

  coverage = torch.zeros(BATCH_SIZE, ENCODER_INPUT_SIZE, device=device)
  context_vector = None
  
  decoder_outputs = []

  for di in range(target_length):
    decoder_output, decoder_hidden, context_vector, attn_weights, coverage = decoder(encoder_outputs, decoder_input, decoder_hidden, coverage, context_vector)
    topv, topi = decoder_output.topk(1)
    decoder_outputs.append(topi.squeeze())
    decoder_input = topi.squeeze()

    newloss = criterion(decoder_output,  target_tensor[:, di])
    loss += newloss

  loss = (loss / target_length).item()
  return loss, decoder_outputs


def evaluateEpoch(encoder, decoder, inputs):
  epoch_len = len(inputs)

  criterion = nn.NLLLoss()

  sample_start = None
  sample_end = None
  tot_loss = 0

  for iter in range(1, epoch_len+1):
    training_pair = inputs[iter-1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss, decoder_outputs = evaluate(input_tensor, target_tensor, encoder, decoder, criterion)
    tot_loss += loss

    if iter == 1:
      sample_start = (loss, decoder_outputs, target_tensor)

    if iter == epoch_len:
      sample_end = (loss, decoder_outputs, target_tensor)

  loss = tot_loss / epoch_len

  return loss, sample_start, sample_end


# ----============= MODEL LOADING =============----
print(Fore.MAGENTA + "\n---- Loading models ----" + Fore.RESET)

encoder = EncoderRNN(
  type_vocab=type_vocab,
  value_vocab=value_vocab,
  hidden_size=HIDDEN_SIZE,
  embedding_size=EMBEDDING_SIZE,
  encoder_input_size=ENCODER_INPUT_SIZE
).to(device)

decoder = AttnDecoderRNN(
  output_vocab_size=len(token_vocab),
  hidden_size=HIDDEN_SIZE,
  embedding_size=EMBEDDING_SIZE,
  batch_size=BATCH_SIZE,
  encoder_input_size=ENCODER_INPUT_SIZE,
  device=device,
).to(device)

# ----============= TRAINING =============----
print(Fore.MAGENTA + "\n---- Training models ----\n" + Fore.RESET)

EPOCHS = 100000
FLAT = 3

PLOT_TIMES = 1000
PRINT_TIMES = 5
BATCH_PRINT_SIZE = 1
SAVE_EVERY = 500
OUTPUT_EVERY = 500

iter_losses = []
train_losses = []
eval_losses = []
prec_loss = 0

start_time = str(datetime.now().strftime("%d.%m_%H.%M"))
output_file = f"{data_path}/output/out-{start_time}.txt"


# with open(output_file, 'w', encoding='utf-8') as outfile: pass

def saveModel(encoder, decoder, epoch):
  torch.save(encoder, f"{data_path}/models/encoder_{start_time}-ep_{epoch}.pt")
  torch.save(decoder, f"{data_path}/models/decoder_{start_time}-ep_{epoch}.pt")

def savePlot(plot, epoch):
  plot.savefig(f"{data_path}/plots/plot_{start_time}.png")

def saveOutput(sample, epoch, extra = ""): #sample = (loss, decoder_outputs, target_tensor)
  loss = sample[0]
  decoder_outputs = torch.stack(sample[1]).view(BATCH_SIZE, -1) # [BATCH, SEQ_LEN]
  target_tensor = sample[2] # [BATCH, SEQ_LEN]

  with open(output_file, 'a', encoding='utf-8') as outfile:
    outfile.write(f"\n================ Epoch: {epoch} | Loss: {loss} | {extra} ================")

    for i in range(min(BATCH_PRINT_SIZE, len(decoder_outputs))):
      predict = token_vocab.batchToSentence(decoder_outputs[i])
      target  = token_vocab.batchToSentence(target_tensor[i]) 

      outfile.write("--------------------\n")
      outfile.write(predict + "\n")
      outfile.write("-.... ↑|predict|↑ ....... ↓|target|↓ ....-\n")
      outfile.write(target + "\n")
    

for epoch in range(1, EPOCHS+1):
  print(Fore.RED + f"----========= EPOCH {epoch}/{EPOCHS} =========----" + Fore.RESET)
  epoch_start = time.time()
  
  random.shuffle(pairs)
  batches = batchPairs(device, pairs, BATCH_SIZE)
  # train_batches, test_batches = split_data(batches)
  train_batches = batches
  test_batches = batches
  print(Fore.GREEN + f"------------------- Inputs loaded -------------------" + Fore.RESET)

  plot_losses, loss_avg = trainEpoch(encoder, decoder, train_batches, print_times=PRINT_TIMES, plot_times=PLOT_TIMES)
  iter_losses += plot_losses
  train_losses.append(loss_avg)

  curr_loss, sample_start, sample_end = evaluateEpoch(encoder, decoder, test_batches)
  eval_losses.append(curr_loss)

  temp_loss = calc_avg_loss(prec_loss, curr_loss)
  if(prec_loss < temp_loss):
    FLAT -= 0
  else:
    FLAT = 3
  
  prec_loss = temp_loss

  print(Fore.GREEN + f"------------------- Finished epoch -------------------")
  print(f"time: {asMinutes(time.time() - epoch_start)}")
  print(f"loss: {curr_loss}, avg loss: {temp_loss}, flat: {FLAT}" + Fore.RESET)

  
  if(FLAT == 0):
    break

  if epoch % OUTPUT_EVERY == 0:
    # saveOutput(sample_start, epoch, "start")
    saveOutput(sample_end, epoch, "end")

  avg_losses = [(train_losses[i] + eval_losses[i]) / 2 for i in range(len(train_losses))]
  loss_diff = [train_losses[i] - eval_losses[i] for i in range(len(train_losses))]

  plot = getPlot([[iter_losses], [train_losses, eval_losses, avg_losses], [loss_diff]])
  savePlot(plot, epoch)
  plt.close('all')

  if epoch % SAVE_EVERY == 0:
    saveModel(encoder, decoder, epoch)



#  TODO:
#- dropout percentuale
#- criterion se serve logsoftmax -> serve.

#- togliere dropout nel test piccolo
#- se funziona tutto attention weights padding

#- cambiare i batch ad ogni epoch

# NOTE: da chiedere
# detach