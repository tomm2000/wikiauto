from tarfile import ENCODING
from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import torch.nn.functional as F

from helpers.helpers import asMsecs, stackTimelog, timelog

class EncoderRNN(nn.Module):
  def __init__(self, type_vocab, value_vocab, hidden_size, embedding_size, encoder_input_size):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size

    self.typeEmbedding = nn.Embedding(len(type_vocab), embedding_size)
    self.valueEmbedding = nn.Embedding(len(value_vocab), embedding_size)
    self.positionEmbedding = nn.Embedding(encoder_input_size, 10)
    
    self.gru = nn.GRU(embedding_size * 2 + 10, self.hidden_size, batch_first=True, bidirectional=True)
    self.dropout1 = nn.Dropout(0.1)
    # self.dropout2 = nn.Dropout(0.5)

    self.outputLinear = nn.Linear(hidden_size*2, hidden_size)
    self.hiddenLinear = nn.Linear(hidden_size*2, hidden_size)

  def forward(self, inputs, hidden):
    E_type_out  = self.typeEmbedding(inputs[0])     # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING_SIZE]
    E_value_out = self.valueEmbedding(inputs[1])    # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING_SIZE]
    E_pos_out   = self.positionEmbedding(inputs[2]) # [BATCH, ENCODER_INPUT_SIZE, 10]

    embedding = torch.cat((E_type_out, E_pos_out, E_value_out), dim=2) # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING_SIZE * 2 + 10]

    output = self.dropout1(embedding) # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING_SIZE * 2 + 10]

    output, hidden = self.gru(output, hidden)
    # output [BATCH, ENCODER_INPUT_SIZE, 2*HIDDEN_SIZE]
    # hidden [2, BATCH, HIDDEN_SIZE] 

    hidden = hidden.view(-1, 2 * self.hidden_size) # [BATCH, 2*HIDDEN_SIZE]

    # linear -> hidden*2 to hidden
    output = self.outputLinear(output) # [BATCH, ENCODER_INPUT_SIZE, HIDDEN_SIZE]
    hidden = self.hiddenLinear(hidden) # [BATCH, HIDDEN_SIZE]

    # relu
    output = F.relu(output)
    hidden = F.relu(hidden)

    return output, hidden

  def initHidden(self, device, batch_size):
    return torch.zeros(2, batch_size, self.hidden_size, device=device)



class AttnCalc(nn.Module):
  def __init__(self, hidden_size, encoder_input_size, batch_size, device):
    super(AttnCalc, self).__init__()
    self.hidden_size = hidden_size
    self.encoder_input_size = encoder_input_size
    self.batch_size = batch_size
    self.device = device

    self.decoderAttnLinear = nn.Linear(hidden_size, hidden_size)

    self.attnLin = nn.Linear(hidden_size, hidden_size)

    self.cvgConv = nn.Conv2d(encoder_input_size, encoder_input_size, (1, hidden_size), stride=1, padding="same")

    self.v = nn.Parameter(nn.init.trunc_normal_(torch.empty(batch_size, hidden_size)), requires_grad=True)

    self.tanhfeatures = nn.Tanh()

  def forward(self, hidden, encoder_outputs, coverage):
    # hidden = [1, BATCH, HIDDEN]
    # encoder_outputs = [BATCH_SIZE, ENCODER_INPUT_SIZE, HIDDEN]
    # coverage = [BATCH, ENCODER_INPUT_SIZE]

    encoder_features = []
    for i in range(self.encoder_input_size):
      tmp_feature_i = self.attnLin(encoder_outputs[:, i, :]) # [BATCH, HIDDEN]
      encoder_features.append(tmp_feature_i)

    encoder_features = torch.stack(encoder_features, dim=1) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

    decoder_features = self.decoderAttnLinear(hidden) # [1, BATCH, HIDDEN]
    decoder_features = decoder_features.view(self.batch_size, 1, self.hidden_size) #- [BATCH, 1, HIDDEN]

    coverage_features = coverage.view(self.batch_size, self.encoder_input_size, 1, -1) # [BATCH, ENCODER_INPUT_SIZE, 1, 1]
    coverage_features = self.cvgConv(coverage_features) # [BATCH_SIZE, ENCODER_INPUT_SIZE, 1, 1]
    coverage_features = coverage_features.view(self.batch_size, self.encoder_input_size, -1) #- [BATCH, ENCODER_INPUT_SIZE, 1]

    # [BATCH, ENCODER_INPUT_SIZE, HIDDEN] + [BATCH, 1, HIDDEN] + [BATCH, ENCODER_INPUT_SIZE, 1]
    attn_features = encoder_features + decoder_features + coverage_features # [BATCH, ENCODER_INPUT_SIZE, HIDDEN]
    attn_features = self.tanhfeatures(attn_features) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

    temp_v = self.v.unsqueeze(2) #- [BATCH, HIDDEN, 1]

    attn_weights = torch.bmm(attn_features, temp_v) # [BATCH, ENCODER_INPUT_SIZE, 1]
    
    attn_weights = F.softmax(attn_weights.squeeze(2), dim=1) #- [BATCH, ENCODER_INPUT_SIZE]

    coverage += attn_weights # [BATCH, ENCODER_INPUT_SIZE]

    context_vector = attn_weights.view(self.batch_size, self.encoder_input_size, 1) * encoder_outputs # [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

    context_vector = torch.sum(context_vector, dim=1) # [BATCH, HIDDEN]

    return context_vector, attn_weights, coverage



class AttnDecoderRNN(nn.Module):
  def __init__(self, output_vocab_size, hidden_size, embedding_size, batch_size, encoder_input_size, device):
    super(AttnDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.output_vocab_size = output_vocab_size
    self.embedding_size = embedding_size
    self.batch_size = batch_size
    self.encoder_input_size = encoder_input_size

    self.calcAttn = AttnCalc(hidden_size, encoder_input_size, batch_size, device).to(device)

    self.embedding = nn.Embedding(self.output_vocab_size, embedding_size)

    self.preOut = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.out = nn.Linear(self.hidden_size, self.output_vocab_size)
    self.newIn = nn.Linear(self.hidden_size + embedding_size, self.hidden_size)

    self.dropout = nn.Dropout(0.1)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    self.tanhout = nn.Tanh()

  def forward(self, encoder_outputs, input, hidden, coverage, context_vector=None):
    embedded = self.embedding(input).view(self.batch_size, -1) # [BATCH, EMBEDDING]
    embedded = self.dropout(embedded) # [BATCH, EMBEDDING]  #NOTE questo è da usare

    # hidden = [1, BATCH, HIDDEN]
    # encoder_outputs = [BATCH, ENCODER_INPUT_SIZE, HIDDEN]
    # coverage = [BATCH, ENCODER_INPUT_SIZE]
    # input= [BATCH, EMBEDDING + HIDDEN]

    if context_vector is None:
      context_vector, _, _ = self.calcAttn(hidden, encoder_outputs, coverage)
    #context_vector = [BATCH, HIDDEN]
    
    new_input = self.newIn(torch.cat((embedded, context_vector), 1)).view(1, self.batch_size, -1) #- [1, BATCH, HIDDEN]

    output, hidden = self.gru(new_input, hidden) # [1, BATCH, HIDDEN]
    output = output.squeeze(0) #- [BATCH, HIDDEN]

    context_vector, attn_weights, coverage = self.calcAttn(hidden, encoder_outputs, coverage)
    #- coverage -> [BATCH, ENCODER_INPUT_SIZE]
    #- context_vector -> [BATCH, HIDDEN]
    #- attn_weights -> [BATCH, ENCODER_INPUT_SIZE]

    output = torch.cat((output, context_vector), 1) # [BATCH, HIDDEN * 2]
    output = self.preOut(output) # [BATCH, HIDDEN]
    output = self.tanhout(output) #- [BATCH, HIDDEN]

    # output = self.dropout2(output) 0.5

    output = self.out(output) # [BATCH, OUTPUT_VOCAB_SIZE]

    output = F.log_softmax(output, dim=1) #- [BATCH, OUTPUT_VOCAB_SIZE]

    return output, hidden, context_vector, attn_weights, coverage

  def initHidden(self, device):
    raise NotImplementedError
    return torch.zeros(1, self.batch_size, self.hidden_size, device=device)