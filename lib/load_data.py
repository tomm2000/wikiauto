from __future__ import unicode_literals, print_function, division
from lib.vocab import vocab, END_TOKEN, START_TOKEN, PADDING_TOKEN, UNKNOWN_TOKEN
from lib.generic import readLines
import json
import torch
from tqdm import tqdm

def prepare_line(table, vocab, phrase_size, apply_end_tnk=False, end_token=END_TOKEN):
  table = table[0:phrase_size]

  while len(table) < phrase_size:
    table.append(vocab.getID(PADDING_TOKEN))

  table = [vocab.getID(el) for el in table]

  if apply_end_tnk:
    table[phrase_size - 1] = vocab.getID(end_token)

  return table


def load_data(device, vocab_size=50000, input_size=30, output_size = 30, pair_amount=1000):
  type_vocab  = vocab(f"data/counts/types.txt", vocab_size)
  value_vocab = vocab(f"data/counts/values.txt", vocab_size)
  token_vocab = vocab(f"data/counts/tokens.txt", vocab_size)
  pairs = []

  lines = readLines(f"data/clean/combined_data_train.json", pair_amount)
  iter = 0

  if pair_amount > len(lines):
    pair_amount = len(lines)
    print("too many pairs requested, new pair amount: ", pair_amount)
  else:
    print(f"selected {pair_amount} pairs out of {len(lines)} available")
  print("------------------------")

  for line in tqdm(lines, desc="loading data: "):
    try:
      json_line = json.loads("{" + line + "}")
    except:
      continue

    article_name = ""

    for n in json_line:
      article_name = n # there is only 1 key per article, the name

    article_data = [
      # types:
      prepare_line(json_line[article_name]["types"], type_vocab, input_size),
      # values:
      prepare_line(json_line[article_name]["values"], value_vocab, input_size),
      # positions:
      [i for i in range(input_size)],
      # tokens (target):
      prepare_line(json_line[article_name]["tokens"], token_vocab, output_size, True)
    ]

    pairs.append(article_data)

    iter += 1
    # print_progress(min(pair_amount, len(lines)), iter, 'loading data', ceil(pair_amount / 10))

  print("------------------------")
  print(f"pairs: {len(pairs)}, input size: {input_size}, output size: {output_size}")

  
  train_size = int(0.8 * len(pairs))

  train_pairs = (
    torch.tensor([pair[0] for pair in pairs[:train_size]], device=device),
    torch.tensor([pair[1] for pair in pairs[:train_size]], device=device),
    torch.tensor([pair[2] for pair in pairs[:train_size]], device=device),
    torch.tensor([pair[3] for pair in pairs[:train_size]], device=device)
  )

  eval_pairs = (
    torch.tensor([pair[0] for pair in pairs[train_size:]], device=device),
    torch.tensor([pair[1] for pair in pairs[train_size:]], device=device),
    torch.tensor([pair[2] for pair in pairs[train_size:]], device=device),
    torch.tensor([pair[3] for pair in pairs[train_size:]], device=device)
  )

  return type_vocab, value_vocab, token_vocab, train_pairs, eval_pairs

def batchPair(pairs, iter, batch_size):
  inputs = (
    torch.stack([pairs[0][i] for i in range(iter, iter + batch_size)]),
    torch.stack([pairs[1][i] for i in range(iter, iter + batch_size)]),
    torch.stack([pairs[2][i] for i in range(iter, iter + batch_size)])
  )

  target = torch.stack([pairs[3][i] for i in range(iter, iter + batch_size)])

  return inputs, target


def getInputSizeAverage():
  count = 0
  sum_types  = 0
  sum_values = 0
  sum_tokens = 0
  lines = readLines('data/clean/combined_data_train.json', -1)

  for line in lines:
    try:
      json_line = json.loads("{" + line + "}")
    except:
      continue

  for article_name in json_line:  # there is only 1 key per article, the name
    sum_types  += len(json_line[article_name]["types"])
    sum_values += len(json_line[article_name]["values"])
    sum_tokens += len(json_line[article_name]["tokens"])
    count += 1

  avg_types  = sum_types  / count
  avg_values = sum_values / count
  avg_tokens = sum_tokens / count

  return avg_types, avg_values, avg_tokens


