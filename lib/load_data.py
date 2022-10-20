import json
from tqdm import tqdm
import torch

from lib.generic import readLines
from lib.vocab import vocab, START_TOKEN, END_TOKEN, PADDING_TOKEN

def load_vocab(folder, vocab_size):
  ''' loads the vocabularies '''
  type_vocab  = vocab(vocab_size)
  value_vocab = vocab(vocab_size)
  token_vocab = vocab(vocab_size)

  token_vocab.addWord(START_TOKEN)
  token_vocab.addWord(END_TOKEN)

  type_vocab.load_file(f'{folder}/types.txt')
  value_vocab.load_file(f'{folder}/values.txt')
  token_vocab.load_file(f'{folder}/tokens.txt')

  return type_vocab, value_vocab, token_vocab


def load_data(dataset, device, input_size, output_size, articles, type_vocab, value_vocab, token_vocab):
  '''
  loads #articles articles from the data folder
  returns
  '''

  # cap the articles to the available articles
  lines = readLines(dataset, size=articles)

  if len(lines) < articles:
    print(f'WARNING: only {len(lines)} articles found in {dataset}')
    articles = len(lines)

  # load the data
  data = []
  
  for line in tqdm(lines, desc="loading data: "):
    line = json.loads(line)

    element = [
      # types:
      torch.tensor(encode_line(line["types"], type_vocab, input_size), device=device),
      # values:
      torch.tensor(encode_line(line["values"], value_vocab, input_size), device=device),
      # positions:
      torch.tensor([i for i in range(input_size)], device=device),
      # tokens (target):
      torch.tensor(encode_line(line["tokens"], token_vocab, output_size, True), device=device)
    ]

    data.append(element)

  return data


def encode_line(data, vocab, size, apply_end_tkn = False):
  ''' encodes a line of data '''
  data = data[0:size]

  if apply_end_tkn:
    if len(data) < size: # if the table is smaller than the phrase size, append the end token
      data.append(END_TOKEN)
    else: # if the table is bigger than the phrase size, replace the last token with the end token
      data[size - 1] = END_TOKEN

  while len(data) < size:
    data.append(PADDING_TOKEN)

  data = [vocab.getID(el) for el in data]

  return data


def make_batch(data, i, batch_size, pad_id):
  '''
  batches the data
  data: type [0], value [1], position [2] and target [3] data
  offset: offset in the data
  '''
  offset = i * batch_size

  inputs = (
    torch.stack([data[b][0] for b in range(offset, offset + batch_size)]),
    torch.stack([data[b][1] for b in range(offset, offset + batch_size)]),
    torch.stack([data[b][2] for b in range(offset, offset + batch_size)])
  )

  target = torch.stack([data[b][3] for b in range(offset, offset + batch_size)])

  attn_mask = torch.where(inputs[0] == pad_id, 0, 1)

  return inputs, target, attn_mask