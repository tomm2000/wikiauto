from __future__ import unicode_literals, print_function, division
from math import ceil, floor
from re import T
from vocab import vocab, END_TOKEN, START_TOKEN, PADDING_TOKEN, UNKNOWN_TOKEN
from helpers import print_progress, readLines
import json

def prepare_line(table, vocab, phrase_size, apply_end_tnk=False, end_token=END_TOKEN):
  table = table[0:phrase_size]

  while len(table) < phrase_size:
      table.append(vocab.getID(PADDING_TOKEN))

  table = [vocab.getID(el) for el in table]

  if apply_end_tnk:
      table[phrase_size - 1] = vocab.getID(end_token)

  return table


def load_data_evaluate(torch, device, type_vocab, value_vocab, batch_size=1, phrase_size=30, amount=1000):
    inputs = []
    batch = []

    POSITIONS = [[i for i in range(phrase_size)] for j in range(batch_size)]

    lines = readLines('data/clean/combined_data_eval.json', amount * batch_size)
    iter = 0

    if amount > len(lines) / batch_size:
        amount = floor(len(lines) / batch_size)
        print("too many pairs requested, new pair amount (articles / batch_size): ", amount)
    else:
        print(f"selected {amount} pairs out of {floor(len(lines) / batch_size)} available")
    print("------------------------")

    for line in lines:
        try:
            json_line = json.loads("{" + line + "}")
        except:
            continue

        for article_name in json_line:  # there is only 1 key per article, the name
            batch.append(json_line[article_name])

        if len(batch) == batch_size:
            # print(batch.keys())

            inputs.append(torch.tensor([
                # types:
                [prepare_line(table["types"], type_vocab, phrase_size) for table in batch],
                # valus:
                [prepare_line(table["values"], value_vocab, phrase_size) for table in batch],
                # positions:
                POSITIONS
            ], device=device))

            batch = []

            iter += 1
            print_progress(min(amount, len(lines)), iter, 'loading data', round(amount / 10))
            if len(inputs) >= amount:
                break

    print("------------------------")
    print(f"pairs: {len(inputs)}, total articles: {len(inputs) * batch_size}")
    print(f"batch size: {batch_size}, phrase size: {phrase_size}")
    print("input shape: ", inputs[0].shape)

    return inputs


def load_data_training(torch, device, vocab_size=50000, batch_size=5, input_size=30, output_size = 30, pair_amount=1000):
  type_vocab = vocab('data/counts/types.txt', vocab_size)
  value_vocab = vocab('data/counts/values.txt', vocab_size)
  token_vocab = vocab('data/counts/tokens.txt', vocab_size)
  pairs = []
  batch = []

  POSITIONS = [[i for i in range(input_size)] for j in range(batch_size)]

  lines = readLines('data/clean/combined_data_train.json', -1)
  iter = 0

  if pair_amount > len(lines) / batch_size:
    pair_amount = floor(len(lines) / batch_size)
    print("too many pairs requested, new pair amount (articles / batch_size): ", pair_amount)
  else:
    print(f"selected {pair_amount} pairs out of {floor(len(lines) / batch_size)} available")
  print("------------------------")

  for line in lines:
    try:
      json_line = json.loads("{" + line + "}")
    except:
      continue

    for article_name in json_line:  # there is only 1 key per article, the name
      batch.append(json_line[article_name])

    if (len(batch) == batch_size):
      # print(batch.keys())

      input_tensor = torch.tensor([
        # types:
        [prepare_line(table["types"], type_vocab, input_size) for table in batch],
        # valus:
        [prepare_line(table["values"], value_vocab, input_size) for table in batch],
        # positions:
        POSITIONS
      ], device=device)

      target_tensor = torch.tensor(
        [prepare_line(table["tokens"], token_vocab, output_size, True) for table in batch],
        device=device
      )

      pairs.append((input_tensor, target_tensor))
      batch = []

      iter += 1
      print_progress(min(pair_amount, len(lines)), iter, 'loading data', ceil(pair_amount / 10))
      if len(pairs) >= pair_amount:
        break

  print("------------------------")
  print(f"pairs: {len(pairs)}, total articles: {len(pairs) * batch_size}")
  print(f"batch size: {batch_size}, input size: {input_size}, output size: {output_size}")
  print("input shape: ", pairs[0][0].shape)
  print("output shape: ", pairs[0][1].shape)

  return type_vocab, value_vocab, token_vocab, pairs


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