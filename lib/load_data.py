from __future__ import unicode_literals, print_function, division
from lib.vocab import vocab, END_TOKEN, START_TOKEN, PADDING_TOKEN, UNKNOWN_TOKEN
from lib.generic import foreach_line, readLines
import json
import torch
from tqdm import tqdm

def encode_line(table, vocab, phrase_size, apply_end_tnk=False):
  table = table[0:phrase_size]

  if apply_end_tnk:
    
    if len(table) < phrase_size: # if the table is smaller than the phrase size, append the end token
      table.append(END_TOKEN)
    else: # if the table is bigger than the phrase size, replace the last token with the end token
      table[phrase_size - 1] = END_TOKEN

  while len(table) < phrase_size:
    table.append(PADDING_TOKEN)
  
  table = [vocab.getID(el) for el in table]

  return table


def load_data(device, vocab_size=50000, input_size=30, output_size = 30, pair_amount=1000):
  type_vocab  = vocab(f"data/counts/types.txt", vocab_size)
  value_vocab = vocab(f"data/counts/values.txt", vocab_size)
  token_vocab = vocab(f"data/counts/tokens.txt", vocab_size)
  pairs = []

  lines = readLines(f"data/clean/dataset.json", pair_amount)

  iter = 0

  if pair_amount > len(lines):
    pair_amount = len(lines)
    print("too many pairs requested, new pair amount: ", pair_amount)
  else:
    print(f"selected {pair_amount} pairs")
  print("------------------------")

  for line in tqdm(lines, desc="loading data: "):
    json_line = json.loads(line)

    article_data = [
      # types:
      encode_line(json_line["types"], type_vocab, input_size),
      # values:
      encode_line(json_line["values"], value_vocab, input_size),
      # positions:
      [i for i in range(input_size)],
      # tokens (target):
      encode_line(json_line["tokens"], token_vocab, output_size, True)
    ]

    pairs.append(article_data)

    iter += 1
    # print_progress(min(pair_amount, len(lines)), iter, 'loading data', ceil(pair_amount / 10))

  print("------------------------")
  print(f"pairs: {len(pairs)}, input size: {input_size}, output size: {output_size}")

  
  train_size = int(0.8 * len(pairs))
  val_size = int(0.1 * len(pairs))
  test_size = int(0.1 * len(pairs))

  print(f"train size: {train_size}, val size: {val_size}, test size: {test_size}")

  train_split = (
    torch.tensor([pair[0] for pair in pairs[:train_size]], device=device),
    torch.tensor([pair[1] for pair in pairs[:train_size]], device=device),
    torch.tensor([pair[2] for pair in pairs[:train_size]], device=device),
    torch.tensor([pair[3] for pair in pairs[:train_size]], device=device)
  )

  eval_split = (
    torch.tensor([pair[0] for pair in pairs[train_size:train_size + val_size]], device=device),
    torch.tensor([pair[1] for pair in pairs[train_size:train_size + val_size]], device=device),
    torch.tensor([pair[2] for pair in pairs[train_size:train_size + val_size]], device=device),
    torch.tensor([pair[3] for pair in pairs[train_size:train_size + val_size]], device=device)
  )

  test_split = (
    torch.tensor([pair[0] for pair in pairs[train_size + val_size:]], device=device),
    torch.tensor([pair[1] for pair in pairs[train_size + val_size:]], device=device),
    torch.tensor([pair[2] for pair in pairs[train_size + val_size:]], device=device),
    torch.tensor([pair[3] for pair in pairs[train_size + val_size:]], device=device)
  )


  return type_vocab, value_vocab, token_vocab, train_split, eval_split, test_split

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
  sum_tokens = 0
  max_types = 0
  max_tokens = 0

  with open('data/clean/dataset.json', 'r', encoding='utf-8') as some_file:
    for line in tqdm(some_file, desc="counting input size: "):
      line = line.lower()
      json_line = json.loads(line)

      sum_types  += len(json_line["types"])
      sum_tokens += len(json_line["tokens"])
      max_types = max(max_types, len(json_line["types"]))
      max_tokens = max(max_tokens, len(json_line["tokens"]))
      count += 1

  avg_types  = sum_types  / count
  avg_tokens = sum_tokens / count

  return avg_types, avg_tokens, max_types, max_tokens


