import torch
import torch.nn as nn
from torch import optim
import time
import torch.nn.functional as F

class EncoderRNN(nn.Module):
  def __init__(self, type_vocab, value_vocab, hidden_size, embedding_size, encoder_input_size):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size

    self.typeEmbedding = nn.Embedding(len(type_vocab), embedding_size)
    self.valueEmbedding = nn.Embedding(len(value_vocab), embedding_size)
    self.positionEmbedding = nn.Embedding(encoder_input_size, 10)
    
    self.gru = nn.GRU(embedding_size * 2 + 10, self.hidden_size, batch_first=True)
    self.dropout1 = nn.Dropout(0.1)
    # self.dropout2 = nn.Dropout(0.5)

  def forward(self, inputs, hidden):
    E_type_out = self.typeEmbedding(inputs[0]) # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING]
    E_value_out = self.valueEmbedding(inputs[1]) # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING]
    E_pos_out = self.positionEmbedding(inputs[2]) # [BATCH,, ENCODER_INPUT_SIZE, EMBEDDING]

    output = torch.cat((E_type_out, E_pos_out, E_value_out), dim=2) # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING * 2 + 10]

    output = self.dropout1(output) # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING * 2 + 10]

    # hidden [BATCH, HIDDEN] ----- output [BATCH, EMBEDDING * 2 + 10]
    output, hidden = self.gru(output, hidden)
    # output [BATCH, ENCODER_INPUT_SIZE, HIDDEN] ----- hidden [1, BATCH, HIDDEN]

    # output = self.dropout2(output)

    return output, hidden

  def initHidden(self, device, batch_size):
    return torch.zeros(1, batch_size, self.hidden_size, device=device)



class AttnCalc(nn.Module):
  def __init__(self, hidden_size, encoder_input_size, batch_size):
    super(AttnCalc, self).__init__()
    self.hidden_size = hidden_size
    self.encoder_input_size = encoder_input_size
    self.batch_size = batch_size

    self.decoderAttnLinear = nn.Linear(hidden_size, hidden_size)

    self.attnConv = nn.Conv2d(encoder_input_size, encoder_input_size, (hidden_size, hidden_size), stride=1, padding="same")
    self.cvgConv = nn.Conv2d(encoder_input_size, encoder_input_size, (1, hidden_size), stride=1, padding="same")

    self.v = nn.Parameter(torch.FloatTensor(batch_size, hidden_size), requires_grad=True)

    self.tanhfeatures = nn.Tanh()

    self.attn_times = [0, 0, 0, 0, 0, 0, 0]

  def forward(self, hidden, encoder_outputs, coverage):
    # hidden = [1, BATCH, HIDDEN]
    # encoder_outputs = [ENCODER_INPUT_SIZE, HIDDEN]
    # coverage = [1, ENCODER_INPUT_SIZE]
    self.attn_times[0] += 1

    # --------------------------------
    start = time.time()

    encoder_features = encoder_outputs.view(self.batch_size, self.encoder_input_size, 1, -1) # [BATCH_SIZE, ENCODER_INPUT_SIZE, 1, HIDDEN]
    encoder_features = self.attnConv(encoder_features) #- [BATCH_SIZE, ENCODER_INPUT_SIZE, 1, HIDDEN]

    self.attn_times[1] = time.time() - start
    # --------------------------------
    # --------------------------------
    start = time.time()

    decoder_features = self.decoderAttnLinear(hidden) # [1, BATCH, HIDDEN]
    decoder_features = decoder_features.view(self.batch_size, 1, 1, -1) #- [BATCH, 1, 1, HIDDEN]
    
    self.attn_times[2] += time.time() - start
    # --------------------------------
    # --------------------------------
    start = time.time()

    coverage_features = coverage.view(self.batch_size, self.encoder_input_size, 1, -1) # [BATCH_SIZE, ENCODER_INPUT_SIZE, 1, 1]
    coverage_features = self.cvgConv(coverage_features) #- [BATCH_SIZE, ENCODER_INPUT_SIZE, 1, 1]


    self.attn_times[3] += time.time() - start
    # --------------------------------
    # --------------------------------
    start = time.time()
    # [1, ENCODER_INPUT_SIZE, 1, HIDDEN] + [BATCH, 1, 1, HIDDEN] + [1, ENCODER_INPUT_SIZE, 1, 1]
    attn_features = encoder_features + decoder_features + coverage_features # [BATCH, ENCODER_INPUT_SIZE, 1, HIDDEN]
    attn_features = attn_features.view(self.batch_size, self.encoder_input_size, -1) # [BATCH, ENCODER_INPUT_SIZE, HIDDEN]
    attn_features = self.tanhfeatures(attn_features) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

    self.attn_times[4] += time.time() - start
    # --------------------------------
    # --------------------------------
    start = time.time()
    temp_v = self.v.unsqueeze(2) #- [BATCH, HIDDEN, 1]
    attn_weights = torch.bmm(attn_features, temp_v) # [BATCH, ENCODER_INPUT_SIZE, 1]
    
    
    self.attn_times[5] += time.time() - start
    # --------------------------------
    # --------------------------------
    start = time.time()
    # attn_weights = torch.sum(attn_weights, dim=2) # [BATCH, ENCODER_INPUT_SIZE]
    attn_weights = F.softmax(attn_weights.squeeze(2), dim=1) #- [BATCH, ENCODER_INPUT_SIZE]


    coverage += attn_weights # [BATCH, ENCODER_INPUT_SIZE]

    context_vector = attn_weights.view(self.batch_size, self.encoder_input_size, 1) * encoder_outputs # [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

    context_vector = torch.sum(context_vector, dim=1) # [BATCH, HIDDEN]

    self.attn_times[6] += time.time() - start
    # --------------------------------

    return context_vector, attn_weights, coverage



class AttnDecoderRNN(nn.Module):
  def __init__(self, output_vocab_size, hidden_size, embedding_size, batch_size, encoder_input_size, device):
    super(AttnDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.output_vocab_size = output_vocab_size
    self.embedding_size = embedding_size
    self.batch_size = batch_size
    self.encoder_input_size = encoder_input_size


    self.calcAttn = AttnCalc(hidden_size, encoder_input_size, batch_size).to(device)

    self.embedding = nn.Embedding(self.output_vocab_size, embedding_size)

    self.preOut = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.out = nn.Linear(self.hidden_size, self.output_vocab_size)
    self.newIn = nn.Linear(self.hidden_size + embedding_size, self.hidden_size)

    self.dropout = nn.Dropout(0.1)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    self.tanhout = nn.Tanh()

    self.decoder_times = [0, 0, 0, 0, 0, 0]

  def forward(self, encoder_outputs, input, hidden, coverage, context_vector=None):
    # --------------------------------
    start = time.time()

    embedded = self.embedding(input).view(self.batch_size, -1) # [BATCH, EMBEDDING]
    embedded = self.dropout(embedded) # [BATCH, EMBEDDING]

    self.decoder_times[0] = time.time() - start
    # --------------------------------


    # --------------------------------
    start = time.time()

    # hidden = [1, BATCH, HIDDEN]
    # encoder_outputs = [BATCH, ENCODER_INPUT_SIZE, HIDDEN]
    # coverage = [BATCH, ENCODER_INPUT_SIZE]

    if context_vector is None:
      context_vector, _, _ = self.calcAttn(hidden, encoder_outputs, coverage)
    #context_vector = [BATCH, HIDDEN]

    self.decoder_times[1] = time.time() - start
    # --------------------------------

    # --------------------------------
    start = time.time()
    
    # input -> [BATCH, EMBEDDING + HIDDEN]
    new_input = self.newIn(torch.cat((embedded, context_vector), 1)).view(1, self.batch_size, -1) #- [1, BATCH, HIDDEN]

    output, hidden = self.gru(new_input, hidden) # [1, BATCH, HIDDEN]
    output = output.squeeze(0) #- [BATCH, HIDDEN]

    self.decoder_times[2] = time.time() - start
    # --------------------------------

    # --------------------------------
    start = time.time()

    context_vector, attn_weights, coverage = self.calcAttn(hidden, encoder_outputs, coverage)
    #- coverage -> [BATCH, ENCODER_INPUT_SIZE]
    #- context_vector -> [BATCH, HIDDEN]
    #- attn_weights -> [BATCH, ENCODER_INPUT_SIZE]

    self.decoder_times[3] = time.time() - start
    # --------------------------------

    # --------------------------------
    start = time.time()

    output = torch.cat((output, context_vector), 1) # [BATCH, HIDDEN * 2]
    output = self.preOut(output) # [BATCH, HIDDEN]
    output = self.tanhout(output) #- [BATCH, HIDDEN] NOTE: da chiedere

    self.decoder_times[4] = time.time() - start
    # --------------------------------

    # output = self.dropout2(output) 0.5

    # --------------------------------
    start = time.time()

    output = self.out(output) # [BATCH, OUTPUT_VOCAB_SIZE]

    output = F.log_softmax(output, dim=1) #- [BATCH, OUTPUT_VOCAB_SIZE]

    self.decoder_times[5] = time.time() - start
    # --------------------------------

    return output, hidden, context_vector, attn_weights, coverage

  def initHidden(self, device):
    return torch.zeros(1, self.batch_size, self.hidden_size, device=device)