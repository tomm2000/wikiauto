# future
from __future__ import unicode_literals, print_function, division
from lib.beam_search import beam_search

# other files
from lib.vocab import END_TOKEN, START_TOKEN
from lib.generic import *
from lib.load_data import batchPair, load_data
from lib.models import EncoderRNN, AttnDecoderRNN
from lib.load_setup import *

# pytorch
import torch
import torch.nn as nn
from torch import optim

# other
import random
import time
from colorama import Fore
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----============= SETUP =============----
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # DEBUG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(Fore.MAGENTA + f"Using device:{Fore.RESET} '{device}'")

SETUP_FOLDER = "setup/test"
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

type_vocab, value_vocab, token_vocab, train_pairs, eval_pairs = load_data(
  device=device,
  vocab_size=VOCAB_SIZE,
  input_size=ENCODER_INPUT_SIZE,
  output_size=DECODER_OUTPUT_SIZE,
  pair_amount=PAIR_AMOUNT,
)

# ----============= TRAINING/EVALUATION FUNCTIONS =============----

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
  encoder_hidden = encoder.initHidden(device, BATCH_SIZE)

  encoder_optimizer.zero_grad(set_to_none=True)
  decoder_optimizer.zero_grad(set_to_none=True)

  target_length = target_tensor.size(1)

  loss = 0

  encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

  decoder_input = torch.tensor([type_vocab.getID(START_TOKEN)] * BATCH_SIZE, device=device)
  decoder_hidden = encoder_hidden

  coverage = torch.zeros(BATCH_SIZE, ENCODER_INPUT_SIZE, device=device)
  context_vector = None

  for di in range(target_length):
    use_teacher_forcing = True if random.random() < TEACHER_FORCIING_RATIO else False

    if use_teacher_forcing: # Teacher forcing: Feed the target as the next input
      decoder_output, decoder_hidden, context_vector, _, coverage = decoder(encoder_outputs, decoder_input, decoder_hidden, coverage, context_vector)
      decoder_input = target_tensor[:, di]
        
    else: # Without teacher forcing: use its own predictions as the next input
      decoder_output, decoder_hidden, context_vector, _, coverage = decoder(encoder_outputs, decoder_input, decoder_hidden, coverage, context_vector)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze()

    newloss = criterion(decoder_output, target_tensor[:, di])

    loss += newloss

  loss = loss / target_length
  perplexity = torch.exp(loss)
  loss.backward()

  # Clip gradient
  nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0, norm_type=2)
  encoder_optimizer.step()

  nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
  decoder_optimizer.step()

  return loss.item(), perplexity.item()


def trainEpoch(encoder, decoder, inputs, plot_times=10000, learning_rate=0.15):
  plot_losses = []
  plot_loss_total = 0  # Reset every plot_every
  loss_total = 0
  tot_perplexity = 0

  epoch_len = math.floor(inputs[0].size(0) / BATCH_SIZE)
  plot_every = max(int(epoch_len / plot_times), 1)

  encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=learning_rate)
  criterion = nn.NLLLoss()

  for iter in tqdm(range(1, epoch_len+1), desc="Training: "):
    # ogni elemento di inputs è una tupla (input, target)
    # ogni valore input è un tensore di dimensione [3, batch, encoder_input_size], deve 3 rappresenta (tipo, valore, posizione)
    # ogni valore target è un tensore di dimensione [batch, decoder_output_size]

    input, target = batchPair(inputs, iter-1, BATCH_SIZE) # (input, target)
    loss, perplexity = train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    plot_loss_total += loss
    loss_total += loss
    tot_perplexity += perplexity

    if iter % plot_every == 0:
      plot_loss_avg = plot_loss_total / plot_every
      plot_losses.append(plot_loss_avg)
      plot_loss_total = 0

  loss_avg = loss_total / epoch_len
  perplexity_avg = tot_perplexity / epoch_len


  return loss_avg, perplexity_avg, plot_losses


def evaluate(input_tensor, target_tensor, encoder, decoder, criterion):

  encoder_hidden = encoder.initHidden(device, BATCH_SIZE)
  target_length = target_tensor.size(1)

  loss = 0

  encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

  # out = beam_search(
  #   decoder=decoder,
  #   beam_size=BATCH_SIZE,
  #   seq_len=DECODER_OUTPUT_SIZE,
  #   encoder_outputs=encoder_outputs,
  #   encoder_hidden=encoder_hidden,
  #   encoder_input_size=ENCODER_INPUT_SIZE,
  #   end_id=token_vocab.getID(END_TOKEN),
  #   start_id=token_vocab.getID(START_TOKEN),
  #   device=device,
  # )

  # raise NotImplementedError

  # loss = criterion(out, target_tensor)
  # # RuntimeError: 0D or 1D target tensor expected, multi-target not supported

  # #----------------------------------------------------#

  # # loss = 0

  # for di in range(target_length):
  #   loss += criterion(out[:, di], target_tensor[:, di])
  #   # RuntimeError: "nll_loss_forward_reduce_cuda_kernel_1d" not implemented for 'Long'

  # loss = (loss / target_length)
  # perplexity = torch.exp(loss)

  # return out
  

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


  loss = (loss / target_length)
  perplexity = torch.exp(loss)


  return loss.item(), perplexity.item(), decoder_outputs


def evaluateEpoch(encoder, decoder, inputs):
  epoch_len = math.floor(inputs[0].size(0) / BATCH_SIZE)

  criterion = nn.NLLLoss()

  sample_start = None
  sample_end = None
  tot_loss = 0
  tot_perplexity = 0

  for iter in range(1, epoch_len+1):
    training_pair = batchPair(inputs, iter-1, BATCH_SIZE)
    # training_pair = inputs[iter-1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss, perplexity, decoder_outputs = evaluate(input_tensor, target_tensor, encoder, decoder, criterion)
    tot_loss += loss
    tot_perplexity += perplexity

    if iter == 1:
      sample_start = (loss, decoder_outputs, target_tensor)

    if iter == epoch_len:
      sample_end = (loss, decoder_outputs, target_tensor)

  loss_avg = tot_loss / epoch_len
  perplexity_avg = tot_perplexity / epoch_len

  return loss_avg, perplexity_avg, sample_start, sample_end


# ----============= MODEL LOADING =============----
print(Fore.MAGENTA + "\n---- Loading models ----" + Fore.RESET)

test_name = "test_1"
epoch_file = f"results/{test_name}/data.txt"

START_EPOCH =  SETUP["epoch"]
flat = SETUP["flat"]
prec_loss = SETUP["prec_loss"]

iter_losses = DATA["iter_losses"]
train_losses = DATA["train_losses"]
eval_losses = DATA["eval_losses"]
train_perplexity = DATA["train_perplexity"]
eval_perplexity = DATA["eval_perplexity"]

model_path = SETUP["model_path"]
result_path = SETUP["result_path"]

encoder = f"{model_path}/encoder_{START_EPOCH}.pt"
decoder = f"{model_path}/decoder_{START_EPOCH}.pt"

print("encoder = " + str(encoder))
print("decoder = " + str(decoder))

try:
  encoder = torch.load(encoder)
  print("loaded old encoder")
except:
  encoder = EncoderRNN(
    type_vocab=type_vocab,
    value_vocab=value_vocab,
    hidden_size=HIDDEN_SIZE,
    embedding_size=EMBEDDING_SIZE,
    encoder_input_size=ENCODER_INPUT_SIZE
  ).to(device)

try:
  decoder = torch.load(decoder)
  print("loaded old decoder")
except:
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

print(Fore.RED + "REMEMBER TO CREATE MODEL OUTPUT FOLDER!!!!" + Fore.RESET)

START_EPOCH += 1
EPOCHS = 100000
PLOT_TIMES = 200
BATCH_PRINT_SIZE = 4
OUTPUT_EVERY = 1

def saveOutput(sample, epoch, extra = ""): #sample = (loss, decoder_outputs, target_tensor)
  loss = sample[0]
  decoder_outputs = torch.stack(sample[1]).view(BATCH_SIZE, -1) # [BATCH, SEQ_LEN]
  target_tensor = sample[2] # [BATCH, SEQ_LEN]

  with open(f"{result_path}/outputs.txt", 'a', encoding='utf-8') as outfile:
    outfile.write(f"\n================ Epoch: {epoch} | Loss: {loss} | {extra} ================")

    for i in range(min(BATCH_PRINT_SIZE, len(decoder_outputs))):
      predict = token_vocab.batchToSentence(decoder_outputs[i])
      target  = token_vocab.batchToSentence(target_tensor[i]) 

      outfile.write("--------------------\n")
      outfile.write(predict + "\n")
      outfile.write("-.... ↑|predict|↑ ....... ↓|target|↓ ....-\n")
      outfile.write(target + "\n")
    

for epoch in range(START_EPOCH, EPOCHS+1):
  print(Fore.RED + f"----========= EPOCH {epoch}/{EPOCHS} =========----" + Fore.RESET)
  epoch_start = time.time()
  
  train_indexes = torch.randperm(train_pairs[0].size(0))
  eval_indexes = torch.randperm(eval_pairs[0].size(0))
  
  print(Fore.GREEN + f"------------------- Inputs loaded -------------------" + Fore.RESET)

  #- train
  loss_avg, perplexity_avg, plot_losses = trainEpoch(encoder, decoder, train_pairs, plot_times=PLOT_TIMES)
  train_losses.append(loss_avg)
  train_perplexity.append(perplexity_avg)

  #- eval
  loss_avg, perplexity_avg, sample_start, sample_end = evaluateEpoch(encoder, decoder, eval_pairs)
  eval_losses.append(loss_avg)
  eval_perplexity.append(perplexity_avg)

  iter_losses += plot_losses
  curr_loss = loss_avg

  temp_loss = calc_avg_loss(prec_loss, curr_loss)
  if(prec_loss < temp_loss):
    flat -= 0
  else:
    flat = 5
  
  prec_loss = temp_loss

  print(Fore.GREEN + f"------------------- Finished epoch -------------------")
  print(f"time: {asMinutes(time.time() - epoch_start)}, loss: {curr_loss}, avg loss: {temp_loss}, flat: {flat}\n" + Fore.RESET)

  #- save model
  encoder_path = f"{model_path}/encoder_{epoch}.pt"
  decoder_path = f"{model_path}/decoder_{epoch}.pt"
  torch.save(encoder, encoder_path)
  torch.save(decoder, decoder_path)

  save_setup(SETUP_FOLDER, 
    epoch=epoch,
    iter_losses=iter_losses,
    train_losses=train_losses,
    eval_losses=eval_losses,
    train_perplexity=train_perplexity,
    eval_perplexity=eval_perplexity,
    prec_loss=prec_loss,
    flat=flat,
  )

  if epoch % OUTPUT_EVERY == 0: saveOutput(sample_end, epoch, "end")

  plot = getPlot([[iter_losses], [train_losses, eval_losses], [train_perplexity, eval_perplexity]])
  plot.savefig(f"{result_path}/plot.png")
  plt.close('all')
  
  #- early stopping
  # if(flat == 0): break


#  TODO:
#- dropout percentuale

#- togliere dropout nel test piccolo
#- se funziona tutto attention weights padding

#- cambiare i batch ad ogni epoch

# NOTE: da chiedere
# detach