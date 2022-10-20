
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.vocab import PADDING_TOKEN, START_TOKEN
import random

class SeqToSeq():
  def __init__(self, encoder, decoder, batch_size, device, token_vocab, input_size, output_size, teacher_forcing_ratio=0.5):
    self.encoder = encoder
    self.decoder = decoder
    self.batch_size = batch_size
    self.device = device
    self.token_vocab = token_vocab
    self.input_size = input_size
    self.output_size = output_size
    self.teacher_forcing_ratio = teacher_forcing_ratio

  def predictNext(self, decoder_input, encoder_outputs, attn_mask, decoder_hidden, coverage, context_vector, criterion = None, target = None):
    '''
    Predicts the next token in the sequence
    -decoder_input: the previous token in the sequence
    -decoder_hidden: the hidden state of the decoder
    -encoder_outputs: the outputs of the encoder
    -criterion: the loss function
    -target: the target token
    '''
    decoder_output, decoder_hidden, context_vector, _, coverage = self.decoder(
      input=decoder_input,
      attn_mask=attn_mask,
      encoder_outputs=encoder_outputs,
      hidden=decoder_hidden,
      coverage=coverage,
      context_vector=context_vector
    )

    loss = None
    if criterion is not None and target is not None:
      loss = criterion(decoder_output, target)
      
    _, decoder_output = decoder_output.topk(1)

    return decoder_output, decoder_hidden, context_vector, coverage, loss

  def predict(self, encoder_input, attn_mask, criterion = None, target = None):
    '''
    Predicts the whole sequence of tokens
    -encoder_input: the 3 inputs to the encoder
    -criterion: the loss function
    -target: the target sequence
    '''
    encoder_hidden = self.encoder.initHidden(self.device, self.batch_size)
    loss = 0

    do_loss = criterion is not None and target is not None

    # runs the encoder
    encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_hidden) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

    decoder_input = torch.tensor([self.token_vocab.getID(START_TOKEN)] * self.batch_size, device=self.device)
    decoder_hidden = encoder_hidden

    coverage = torch.zeros(self.batch_size, self.input_size, device=self.device)
    context_vector = None

    # runs the decoder
    decoder_outputs = []

    for di in range(self.output_size):
      use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

      target_token = None
      if do_loss: target_token = target[:, di]

      if (target_token == self.token_vocab.getID(PADDING_TOKEN)).all():
        break

      decoder_output, decoder_hidden, context_vector, coverage, new_loss = self.predictNext(
        decoder_input=decoder_input,
        encoder_outputs=encoder_outputs,
        attn_mask=attn_mask,
        decoder_hidden=decoder_hidden,
        coverage=coverage,
        context_vector=context_vector,
        criterion=criterion,
        target=target_token
      )
      decoder_outputs.append(decoder_output)

      if use_teacher_forcing:
        decoder_input = target[:, di]
      else:
        decoder_input = decoder_output.squeeze().detach()

      if do_loss:
        loss += new_loss

    # returns the loss and the predicted sequence
    if do_loss:
      loss = torch.sum(loss, dim=0)
      loss = loss / self.batch_size
      loss = loss / self.output_size

      return decoder_outputs, loss

    return decoder_outputs, None

  def train(self, encoder_input, attn_mask, encoder_optimizer, decoder_optimizer, criterion, target):
    '''
    Trains the model
    -encoder_input: the 3 inputs to the encoder
    -criterion: the loss function
    -target: the target sequence
    -encoder_optimizer: the optimizer for the encoder
    -decoder_optimizer: the optimizer for the decoder
    '''
    self.encoder.train()
    self.decoder.train()

    encoder_optimizer.zero_grad(set_to_none=True)
    decoder_optimizer.zero_grad(set_to_none=True)

    decoder_outputs, loss = self.predict(
      encoder_input=encoder_input,
      attn_mask=attn_mask,
      criterion=criterion,
      target=target
    )

    loss.backward()

    # Clip gradient
    nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=5.0, norm_type=2)
    encoder_optimizer.step()

    nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=5.0)
    decoder_optimizer.step()

    return decoder_outputs, loss

  def set_eval(self):
    self.encoder.eval()
    self.decoder.eval()

  def set_train(self):
    self.encoder.train()
    self.decoder.train()

class EncoderRNN(nn.Module):
  def __init__(self, type_vocab, value_vocab, hidden_size, embedding_size, input_size):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size

    self.typeEmbedding = nn.Embedding(len(type_vocab), embedding_size)
    self.valueEmbedding = nn.Embedding(len(value_vocab), embedding_size)
    self.positionEmbedding = nn.Embedding(input_size, 10)
    
    self.gru = nn.GRU(embedding_size * 2 + 10, self.hidden_size, batch_first=True, bidirectional=True)
    self.dropout = nn.Dropout(0.1)

    self.outputLinear = nn.Linear(hidden_size*2, hidden_size)
    self.hiddenLinear = nn.Linear(hidden_size*2, hidden_size)

  def forward(self, inputs, hidden):

    E_type_out  = self.typeEmbedding(inputs[0])     # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING_SIZE]
    E_value_out = self.valueEmbedding(inputs[1])    # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING_SIZE]
    E_pos_out   = self.positionEmbedding(inputs[2]) # [BATCH, ENCODER_INPUT_SIZE, 10]

    embedding = torch.cat((E_type_out, E_pos_out, E_value_out), dim=2) # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING_SIZE * 2 + 10]

    output = self.dropout(embedding) # [BATCH, ENCODER_INPUT_SIZE, EMBEDDING_SIZE * 2 + 10]
    output, hidden = self.gru(output, hidden) # [BATCH, ENCODER_INPUT_SIZE, 2*HIDDEN_SIZE] | [2, BATCH, HIDDEN_SIZE]
    hidden = hidden.view(1, -1, 2 * self.hidden_size) # [1, BATCH, 2*HIDDEN_SIZE]

    output = self.outputLinear(output) # [BATCH, ENCODER_INPUT_SIZE, HIDDEN_SIZE]
    hidden = self.hiddenLinear(hidden) # [1, BATCH, HIDDEN_SIZE]

    output = F.relu(output) # [BATCH, ENCODER_INPUT_SIZE, HIDDEN_SIZE]
    hidden = F.relu(hidden) # [1, BATCH, HIDDEN_SIZE]

    return output, hidden

  def initHidden(self, device, batch_size):
    return torch.zeros(2, batch_size, self.hidden_size, device=device)

class AttnCalc(nn.Module):
  def __init__(self, hidden_size, input_size, batch_size):
    super(AttnCalc, self).__init__()
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.batch_size = batch_size

    self.decoderAttnLinear = nn.Linear(hidden_size, hidden_size)
    self.attnLin = nn.Linear(hidden_size, hidden_size)
    self.cvgConv = nn.Conv2d(input_size, input_size, (1, hidden_size), stride=1, padding="same")

    self.v = nn.Parameter(nn.init.trunc_normal_(torch.empty(batch_size, hidden_size)), requires_grad=True)

  def forward(self, encoder_outputs, attn_mask, hidden, coverage):
    # hidden = [1, BATCH, HIDDEN]
    # encoder_outputs = [BATCH_SIZE, ENCODER_INPUT_SIZE, HIDDEN]
    # coverage = [BATCH, ENCODER_INPUT_SIZE]

    encoder_features = []
    for i in range(self.input_size):
      tmp_feature_i = self.attnLin(encoder_outputs[:, i, :]) # [BATCH, HIDDEN]
      encoder_features.append(tmp_feature_i)

    encoder_features = torch.stack(encoder_features, dim=1) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]


    decoder_features = self.decoderAttnLinear(hidden) # [1, BATCH, HIDDEN]
    decoder_features = decoder_features.view(self.batch_size, 1, self.hidden_size) #- [BATCH, 1, HIDDEN]

    coverage_features = coverage.view(self.batch_size, self.input_size, 1, -1) # [BATCH, ENCODER_INPUT_SIZE, 1, 1]
    coverage_features = self.cvgConv(coverage_features) # [BATCH_SIZE, ENCODER_INPUT_SIZE, 1, 1]
    coverage_features = coverage_features.view(self.batch_size, self.input_size, -1) #- [BATCH, ENCODER_INPUT_SIZE, 1]

    # [BATCH, ENCODER_INPUT_SIZE, HIDDEN] + [BATCH, 1, HIDDEN] + [BATCH, ENCODER_INPUT_SIZE, 1]
    attn_features = encoder_features + decoder_features + coverage_features # [BATCH, ENCODER_INPUT_SIZE, HIDDEN]
    attn_features = torch.tanh(attn_features) #- [BATCH, ENCODER_INPUT_SIZE, HIDDEN]

    temp_v = self.v.unsqueeze(2) #- [BATCH, HIDDEN, 1]

    attn_weights = torch.bmm(attn_features, temp_v).squeeze(2) # [BATCH, ENCODER_INPUT_SIZE]
    attn_weights = torch.where(attn_mask == 1, attn_weights, -torch.inf) # [BATCH, ENCODER_INPUT_SIZE]
    attn_weights = F.softmax(attn_weights, dim=1) #- [BATCH, ENCODER_INPUT_SIZE]

    coverage += attn_weights # [BATCH, ENCODER_INPUT_SIZE]

    context_vector = attn_weights.view(self.batch_size, self.input_size, 1) * encoder_outputs # [BATCH, ENCODER_INPUT_SIZE, HIDDEN]
    context_vector = torch.sum(context_vector, dim=1) # [BATCH, HIDDEN]

    return context_vector, attn_weights, coverage



class DecoderRNN(nn.Module):
  def __init__(self, output_vocab_size, hidden_size, embedding_size, batch_size, input_size, device):
    super(DecoderRNN, self).__init__()
    self.batch_size = batch_size

    self.calcAttn = AttnCalc(hidden_size, input_size, batch_size).to(device)
    self.embedding = nn.Embedding(output_vocab_size, embedding_size)

    self.preOut = nn.Linear(hidden_size * 2, hidden_size)
    self.out = nn.Linear(hidden_size,output_vocab_size)
    self.newIn = nn.Linear(hidden_size + embedding_size, hidden_size)

    self.dropout = nn.Dropout(0.1)
    self.gru = nn.GRU(hidden_size, hidden_size)

  def forward(self, input, attn_mask, encoder_outputs, hidden, coverage, context_vector=None):
    # hidden = [1, BATCH, HIDDEN]
    # encoder_outputs = [BATCH, ENCODER_INPUT_SIZE, HIDDEN]
    # coverage = [BATCH, ENCODER_INPUT_SIZE]
    # input= [BATCH, EMBEDDING + HIDDEN]

    embedded = self.embedding(input).view(self.batch_size, -1) # [BATCH, EMBEDDING]
    embedded = self.dropout(embedded) # [BATCH, EMBEDDING]  #NOTE questo è da usare

    if context_vector is None:
      context_vector, _, _ = self.calcAttn(
        encoder_outputs=encoder_outputs,
        attn_mask=attn_mask,
        hidden=hidden,
        coverage=coverage
      ) # [BATCH, HIDDEN]
    
    new_input = self.newIn(torch.cat((embedded, context_vector), 1)).view(1, self.batch_size, -1) #- [1, BATCH, HIDDEN]

    output, hidden = self.gru(new_input, hidden) # [1, BATCH, HIDDEN]
    output = output.squeeze(0) #- [BATCH, HIDDEN]

    context_vector, attn_weights, coverage = self.calcAttn(
      encoder_outputs=encoder_outputs,
      attn_mask=attn_mask,
      hidden=hidden,
      coverage=coverage
    )
    #- coverage -> [BATCH, ENCODER_INPUT_SIZE]
    #- context_vector -> [BATCH, HIDDEN]
    #- attn_weights -> [BATCH, ENCODER_INPUT_SIZE]

    output = torch.cat((output, context_vector), 1) # [BATCH, HIDDEN * 2]
    output = self.preOut(output) # [BATCH, HIDDEN]

    output = self.out(output) # [BATCH, OUTPUT_VOCAB_SIZE]

    output = F.log_softmax(output, dim=1) #- [BATCH, OUTPUT_VOCAB_SIZE]

    return output, hidden, context_vector, attn_weights, coverage

  def initHidden(self):
    raise Exception("initialize with encoder hidden state")