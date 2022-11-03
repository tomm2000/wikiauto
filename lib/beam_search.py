import time
import numpy as np
import torch

from lib.generic import timelog

# classe per rappresentare un cammino nella beam search
# NON MODIFICATO
class Hypothesis(object):

  def __init__(self, tokens, log_probs, state, coverage):

    self._tokens = tokens
    self._log_probs = log_probs
    self._state = state
    self._coverage = coverage

  def extend(self, token, log_prob, state, coverage):
    return Hypothesis(self._tokens+[token], self._log_probs+[log_prob], state, coverage)

  @property
  def last_token(self):
    return self._tokens[-1]

  @property
  def log_prob(self):
    return sum(self._log_probs)

  @property
  def length(self):
    return len(self._tokens)

  @property
  def avg_log_prob(self):
    return self.log_prob / self.length

  @property
  def tokens(self):
    return self._tokens


def beam_search(decoder, batch_size, beam_size, seq_len, encoder_outputs, attn_mask, encoder_hidden, encoder_input_size, end_id, start_id, device):
  # beam_size: inizializzato a batch size
  # encoder_input_size: quanti input ha l'encoder
  # end_id: id del token di fine frase
  # start_id: id del token di inizio frase
  MIN_SEQ_LEN = 35

  step = 0
  results = []

  decoder_hidden = encoder_hidden # [1, BATCH_SIZE, HIDDEN_SIZE]
  coverage = torch.zeros(batch_size, encoder_input_size, device=device) # [BATCH_SIZE, ENCODER_INPUT_SIZE]
  context_vector = None # [BATCH_SIZE, HIDDEN_SIZE]
  
  hyps = np.array([Hypothesis([start_id], [0.0], decoder_hidden[:, i, :], coverage[i, :]) for i in range(batch_size)])  # [BEAM_SIZE]


  while step < seq_len and len(results) < beam_size:
    # prepariamo l'input per la rete neurale
    tokens = torch.tensor([h.last_token for h in hyps], device=device) # [BEAM_SIZE]

    # fill tokens up to batch size length
    if len(tokens) < batch_size:
      tokens = torch.cat((tokens, torch.tensor([end_id] * (batch_size - len(tokens)), device=device)))

    tokens = tokens.unsqueeze(0) # [1, BEAM_SIZE]

    all_hyps = []

    for i, hyp in enumerate(hyps):
      decoder_hidden[:, i, :] = hyp._state
      coverage[i, :] = hyp._coverage

    probs, decoder_hidden, context_vector, _, coverage = decoder(encoder_outputs, tokens, attn_mask, decoder_hidden, coverage, context_vector)  

    # per ogni cammino nella beam search
    for i, h in enumerate(hyps):
      # generiamo la distribuzione di probabilita'
      # pere prendere la probabilita della ipotesi al tempo e step corrente
      # usate [i,step]. Copiate da model/make_name l'estrazione delle probabilita
      # in questo caso, non serve prendere il character corrente (lo faremo alla fine)

      indexes = probs.topk(beam_size * 2, dim=1)

      # scegliamo 2 * beam_size possibili espansioni dei cammini
      for j in range(beam_size*2):
        # aggiungere ad all_hyps l'estensione delle ipotesi
        # con il j-esimo indice e la sua probabilita'
        # usate h.extend() per farlo
        all_hyps.append(h.extend(indexes[1][i, j], indexes[0][i, j], decoder_hidden[:, i, :], coverage[i, :]))

    # teniamo solo beam_size cammini migliori
    hyps = []
    for h in sort_hyps(all_hyps):
      if h.last_token == end_id:
        if step >= MIN_SEQ_LEN:
          results.append(h)
      else:
        hyps.append(h)

      if len(hyps) == beam_size or len(results) == beam_size:
          break

    step += 1


  if len(results) == 0:
      results = hyps

  hyps_sorted = sort_hyps(results)

  output = torch.tensor(hyps_sorted[0].tokens[1:], device=device)

  return output


def sort_hyps(hyps):
  # ritornate le ipotesi in ordine descrescente di probabilita media
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
