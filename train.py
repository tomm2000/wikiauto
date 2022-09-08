# future
from __future__ import unicode_literals, print_function, division

# other files
from helpers.vocab import START_TOKEN
from helpers.helpers import *
from helpers.load_data import load_data_training
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"device: {device}")

print(Fore.MAGENTA + f"Using device:{Fore.RESET} '{device}'")

base_path = "data"
# device = "cpu"

ENCODER_INPUT_SIZE = 40 # dimensione dell'input dell'encoder (numero di triple tipo-valore-posizione in input)
DECODER_OUTPUT_SIZE = 40 # dimensione dell'output del decoder (lunghezza della frase in output)
BATCH_SIZE = 3
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 128

print(Fore.MAGENTA + "\n---- Loading data ----" + Fore.RESET)

type_vocab, value_vocab, token_vocab, pairs = load_data_training(
  torch=torch,
  device=device,
  vocab_size=5000,
  batch_size=BATCH_SIZE,
  input_size=ENCODER_INPUT_SIZE,
  output_size=DECODER_OUTPUT_SIZE,
  pair_amount=100,
  path=base_path
)

def split_data(pairs, train_size=0.8):
  train_size = int(train_size * len(pairs))
  train_pairs = pairs[:train_size]
  test_pairs = pairs[train_size:]
  return train_pairs, test_pairs

train_pairs, test_pairs = split_data(pairs)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

  start = time.time() #-----------------------------------------------
  encoder_hidden = encoder.initHidden(device, BATCH_SIZE)

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  input_length = input_tensor.size(2) # == PHRASE_SIZE
  target_length = target_tensor.size(1)

  print(Fore.GREEN + f"Initialization: {asMsecs(time.time() - start)}" + Fore.RESET) #-----------------------------------------------
  start = time.time() #-----------------------------------------------

  loss = 0

  encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

  decoder_input = torch.tensor([type_vocab.getID(START_TOKEN) for _ in range(BATCH_SIZE)], device=device)
  decoder_hidden = encoder_hidden

  coverage = torch.zeros(BATCH_SIZE, ENCODER_INPUT_SIZE, device=device)
  context_vector = None

  print(Fore.GREEN + f"Encoder: {asMsecs(time.time() - start)}" + Fore.RESET) #-----------------------------------------------
  start = time.time() #-----------------------------------------------

  for di in range(target_length):
    start_cycle = time.time()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # use_teacher_forcing = True

    current_target = target_tensor[:, di] # [BATCH]

    if use_teacher_forcing:
      # Teacher forcing: Feed the target as the next input
      decoder_output, decoder_hidden, context_vector, attn_weights, coverage = decoder(encoder_outputs, decoder_input, decoder_hidden, coverage, context_vector)
      decoder_input = current_target
        
    else:
      # Without teacher forcing: use its own predictions as the next input
      decoder_output, decoder_hidden, context_vector, attn_weights, coverage = decoder(encoder_outputs, decoder_input, decoder_hidden, coverage, context_vector)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze()

    loss += criterion(decoder_output, current_target)

    cycle_time = time.time() - start_cycle
    color = Fore.WHITE if cycle_time < 0.2 else Fore.RED
    
    print(color + f"iter time {di}: {asMsecs(cycle_time)}" + Fore.RESET)

    times = ""

    for i in range(len(decoder.decoder_times)):
      color2 = Fore.WHITE if decoder.decoder_times[i] < 0.2 else Fore.RED
      times += color2 + f"[{i}]: {asMsecs(decoder.decoder_times[i])} | " + Fore.RESET

    print(f"decoder times: {times}")

    print("------------------")

  print(Fore.GREEN + f"Decoder: {asMsecs(time.time() - start)}") #-----------------------------------------------
  start = time.time() #-----------------------------------------------

  # raise Exception("stop")
  
  start = time.time()

  loss = loss / target_length
  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  print(Fore.GREEN + f"Backward: {asMsecs(time.time() - start)}") #-----------------------------------------------


  return loss.item()


def trainEpoch(encoder, decoder, inputs, print_times=10, plot_times=10000, learning_rate=5e-5):
  start = time.time()
  plot_losses = []
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every
  epoch_len = len(inputs)
  plot_every = max(int(epoch_len / plot_times), 1)
  print_every = max(int(epoch_len / print_times), 1)

  encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
  criterion = nn.NLLLoss()

  for iter in range(1, epoch_len+1):
    # ogni elemento di inputs è una tupla (input, target)
    # ogni valore input è un tensore di dimensione [3, batch, encoder_input_size], deve 3 rappresenta (tipo, valore, posizione)
    # ogni valore target è un tensore di dimensione [batch, decoder_output_size]

    training_pair = inputs[iter-1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    start_time = time.time()

    loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    print(f"iter {iter} time: {timeSince(start_time, 1)}")

    print_loss_total += loss
    plot_loss_total += loss

    if iter % print_every == 0:
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0
      print(f"{timeSince(start, iter / epoch_len+1)} ({iter} {iter / (epoch_len+1) * 100:.2f}%) {print_loss_avg:.4f}")

    if iter % print_every == 0:
      plot_loss_avg = plot_loss_total / plot_every
      plot_losses.append(plot_loss_avg)
      plot_loss_total = 0

  showPlot(plot_losses)
  return getPlot(plot_losses)


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



print(Fore.MAGENTA + "\n---- Training models ----\n" + Fore.RESET)

EPOCHS = 1
FLAT = 3

PLOT_TIMES = 1000
PRINT_TIMES = 5
BATCH_PRINT_SIZE = 5
SAVE_MODEL_EVERY = 1
SAVE_PLOT_EVERY = 1

start_time = str(datetime.now().strftime("%d.%m_%H.%M"))
output_file = f"{base_path}/output/out-{start_time}.txt"

# with open(output_file, 'w', encoding='utf-8') as outfile: pass

prec_loss = 0

def saveModel(encoder, decoder, epoch):
  torch.save(encoder, f"{base_path}/models/encoder_{start_time}-ep_{epoch}-iters.pt")
  torch.save(decoder, f"{base_path}/models/decoder_{start_time}-ep_{epoch}-iters.pt")

def savePlot(plot, epoch):
  plot.savefig(f"{base_path}/plots/plot_{start_time}-ep_{epoch}-iters.png")

def saveOutput(output, target, epoch):
  with open(output_file, 'a', encoding='utf-8') as outfile:
    for i in range(min(BATCH_PRINT_SIZE, len(output))):
      outfile.write("----------------------\n")
      predict = ""
      target = ""
      for word in output[i]:
        predict += str(word) + " "
      outfile.write(predict + "\n")
      # outfile.write("-.... ↑|predict|↑ ....... ↓|target|↓ ....-\n")
      # for word in pairs[0][1][i]:
      #   target += token_vocab.getWord(word.item()) + " "
      # outfile.write(target + "\n")
    

for epoch in range(1, EPOCHS+1):
  print(f"----========= EPOCH {epoch}/{EPOCHS}=========----")
  epoch_start = time.time()
  
  random.shuffle(pairs)
  plot = trainEpoch(encoder, decoder, train_pairs, print_times=PRINT_TIMES, plot_times=PLOT_TIMES)
  print(f"------------------- Trained -------------------")
  # curr_loss, sample_start, sample_end = evaluateEpoch(encoder, decoder, test_pairs)

  # temp_loss = calc_avg_loss(prec_loss, curr_loss)
  # if(prec_loss < temp_loss):
  #   FLAT -= 1
  # else:
  #   FLAT = 3
  
  # if(FLAT == 0):
  #   break
  
  # prec_loss = temp_loss

  print(f"------------------- Finished epoch -------------------")
  print(f"time: {int((time.time() - epoch_start)/60)}min")
  # print(f"loss: {curr_loss}, avg loss: {temp_loss}, flat: {FLAT}")

  # saveOutput(sample_start[0], sample_start[1], epoch)
  # saveOutput(sample_end[0], sample_end[1], epoch)

  # if epoch % SAVE_PLOT_EVERY == 0:
  #   savePlot(plot, epoch, epoch)

  # if epoch % SAVE_MODEL_EVERY == 0:
  #   saveModel(encoder, decoder, epoch, epoch)


  


#  TODO:
#- dropout percentuale
#- criterion se serve logsoftmax

#- togliere dropout nel test piccolo
#- se funziona tutto attention weights padding
