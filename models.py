from torch import nn
import torch
import torch.nn.functional as F

# ENCODER
class EncoderRNN(nn.Module):
  def __init__(self, type_vocab, value_vocab, hidden_size, embedding_size, phrase_size, batch_size, device):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size

    self.batch_size = batch_size
    self.device = device

    self.typeEmbedding = nn.Embedding(len(type_vocab), embedding_size, device=device)
    self.valueEmbedding = nn.Embedding(len(value_vocab), embedding_size, device=device)
    self.positionEmbedding = nn.Embedding(phrase_size, 10, device=device)
    
    self.gru = nn.GRU(embedding_size * 2 + 10, self.hidden_size)

  def forward(self, inputs, hidden):
    E_type_out = self.typeEmbedding(inputs[0])
    E_value_out = self.valueEmbedding(inputs[1])
    E_pos_out = self.positionEmbedding(inputs[2])

    output = torch.cat((E_type_out, E_pos_out, E_value_out), dim=1).view(1, self.batch_size, -1)

    output, hidden = self.gru(output, hidden)
    return output, hidden

  def initHidden(self):
    return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)



# DECODER
class AttnDecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size, phrase_size, batch_size, device, dropout_p=0.1):
    super(AttnDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.dropout_p = dropout_p
    self.max_length = phrase_size
    self.batch_size = batch_size

    self.embedding = nn.Embedding(self.output_size, self.hidden_size)
    self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
    self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)
    self.out = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, input, hidden, encoder_outputs):
    embedded = self.embedding(input).view(1, self.batch_size, -1)
    embedded = self.dropout(embedded)

    attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
    attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)

    output = F.relu(output)
    output, hidden = self.gru(output, hidden)

    output = F.log_softmax(self.out(output[0]), dim=1)
    return output, hidden, attn_weights

  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size, device=self.device)
