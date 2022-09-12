from __future__ import unicode_literals, print_function, division
from math import ceil, floor
from helpers.vocab import vocab, END_TOKEN, START_TOKEN, PADDING_TOKEN, UNKNOWN_TOKEN
from helpers.helpers import print_progress, readLines
import json
import torch

def prepare_line(table, vocab, phrase_size, apply_end_tnk=False, end_token=END_TOKEN):
  table = table[0:phrase_size]

  while len(table) < phrase_size:
    table.append(vocab.getID(PADDING_TOKEN))

  table = [vocab.getID(el) for el in table]

  if apply_end_tnk:
    table[phrase_size - 1] = vocab.getID(end_token)

  return table


def load_data_training(vocab_size=50000, input_size=30, output_size = 30, pair_amount=1000, path = "data"):
  type_vocab  = vocab(f"{path}/counts/types.txt", vocab_size)
  value_vocab = vocab(f"{path}/counts/values.txt", vocab_size)
  token_vocab = vocab(f"{path}/counts/tokens.txt", vocab_size)
  pairs = []

  lines = readLines(f"{path}/clean/combined_data_train.json", -1)
  iter = 0

  if pair_amount > len(lines):
    pair_amount = len(lines)
    print("too many pairs requested, new pair amount: ", pair_amount)
  else:
    print(f"selected {pair_amount} pairs out of {len(lines)} available")
  print("------------------------")

  for line in lines:
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
    print_progress(min(pair_amount, len(lines)), iter, 'loading data', ceil(pair_amount / 10))
    if len(pairs) >= pair_amount:
      break

  print("------------------------")
  print(f"pairs: {len(pairs)}, input size: {input_size}, output size: {output_size}")

  return type_vocab, value_vocab, token_vocab, pairs


def batchPairs(device, pairs, batch_size):
  batches = []

  len_pairs = len(pairs) - (len(pairs) % batch_size)

  for i in range(0, len_pairs, batch_size):
    inputs = (
      torch.tensor([pair[0] for pair in pairs[i:i + batch_size]], device=device),
      torch.tensor([pair[1] for pair in pairs[i:i + batch_size]], device=device),
      torch.tensor([pair[2] for pair in pairs[i:i + batch_size]], device=device)
    )

    target = torch.tensor([pair[3] for pair in pairs[i:i + batch_size]], device=device)

    batches.append((inputs, target))

  print(f"{len_pairs} pairs grouped in {len(batches)} batches of size {batch_size}")
  print(f"input size: 3 x {batches[0][0][0].shape}")
  print(f"target size: 1 x {batches[0][1].shape}")

  return batches


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


