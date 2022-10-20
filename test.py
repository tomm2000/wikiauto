from random import randint
import torch

from lib.load_data import load_data, load_vocab, make_batch
from lib.vocab import END_TOKEN, PADDING_TOKEN, START_TOKEN, UNKNOWN_TOKEN

# --------------------- SETUP ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- VOCABS ---------------------
print("TEST 1: Creating vocabularies")

VOCAB_SIZE = 1234

type_vocab, value_vocab, token_vocab = load_vocab(
  folder="data/counts/",
  vocab_size=VOCAB_SIZE
)

assert type_vocab.getID(UNKNOWN_TOKEN) == 0, f"UNKNOWN_TOKEN should be 0, instead it is {type_vocab.getID(UNKNOWN_TOKEN)}"
assert type_vocab.getID(PADDING_TOKEN) == 1, f"PADDING_TOKEN should be 1, instead it is {type_vocab.getID(PADDING_TOKEN)}"

assert value_vocab.getID(UNKNOWN_TOKEN) == 0, f"UNKNOWN_TOKEN should be 0, instead it is {value_vocab.getID(UNKNOWN_TOKEN)}"
assert value_vocab.getID(PADDING_TOKEN) == 1, f"PADDING_TOKEN should be 1, instead it is {value_vocab.getID(PADDING_TOKEN)}"

assert token_vocab.getID(UNKNOWN_TOKEN) == 0, f"UNKNOWN_TOKEN should be 0, instead it is {token_vocab.getID(UNKNOWN_TOKEN)}"
assert token_vocab.getID(PADDING_TOKEN) == 1, f"PADDING_TOKEN should be 1, instead it is {token_vocab.getID(PADDING_TOKEN)}"
assert token_vocab.getID(START_TOKEN)   == 2, f"START_TOKEN should be 2, instead it is {token_vocab.getID(START_TOKEN)}"
assert token_vocab.getID(END_TOKEN)     == 3, f"END_TOKEN should be 3, instead it is {token_vocab.getID(END_TOKEN)}"

assert len(type_vocab)  <= VOCAB_SIZE, f"Type vocab size should be 1234, instead it is {len(type_vocab)}"
assert len(value_vocab) <= VOCAB_SIZE, f"Value vocab size should be 1234, instead it is {len(value_vocab)}"
assert len(token_vocab) <= VOCAB_SIZE, f"Token vocab size should be 1234, instead it is {len(token_vocab)}"

print("TEST 1: PASSED")

# --------------------- DATA ---------------------
print("TEST 2: Loading data")
INPUT_SIZE = 25
OUTPUT_SIZE = 25
ARTICLES = 100
BATCH_SIZE = 3

data = load_data("data/dataset/train.json", device, INPUT_SIZE, OUTPUT_SIZE, ARTICLES, type_vocab, value_vocab, token_vocab)

assert len(data) == ARTICLES, f"Data should have 100 examples, instead it has {len(data)}"

assert len(data[0]) == 4, f"Data should have 4 elements, instead it has {len(data[0])}"

assert len(data[0][0]) == INPUT_SIZE, f"Data should have 10 types, instead it has {len(data[0][0])}"
assert len(data[0][1]) == INPUT_SIZE, f"Data should have 10 values, instead it has {len(data[0][0])}"
assert len(data[0][3]) == OUTPUT_SIZE, f"Data should have 10 values, instead it has {len(data[0][0])}"

for i in range(len(data[0][2])):
  assert data[0][2][i] == i, f"Position {i} should be {i}, instead it is {data[0][2][i]}"

end_found = False

for i in range(len(data[0][3])):
  if end_found:
    assert data[0][3][i] == token_vocab.getID(PADDING_TOKEN), f"Tokens after END should be PAD (1), {i} instead is {token_vocab.getID(data[0][3][i])}"
  if data[0][3][i] == token_vocab.getID(END_TOKEN):
    end_found = True

inputs1, targets1, mask1 = make_batch(data, 0, BATCH_SIZE, type_vocab.getID(PADDING_TOKEN))

assert len(inputs1) == BATCH_SIZE, f"Batch should have 4 elements, instead it has {len(inputs1)}"

assert (inputs1[0][0] == data[0][0]).all(), f"Batch should have the same data as the first example ({data[0][0]}), instead it has {inputs1[0][0]}"
assert (inputs1[1][0] == data[0][1]).all(), f"Batch should have the same data as the first example ({data[0][1]}), instead it has {inputs1[1][0]}"
assert (inputs1[2][0] == data[0][2]).all(), f"Batch should have the same data as the first example ({data[0][2]}), instead it has {inputs1[2][0]}"

assert (targets1[0] == data[0][3]).all(), f"Batch should have the same data as the first example ({data[0]}), instead it has {inputs1[0]}"

inputs2, targets2, mask2 = make_batch(data, 1, BATCH_SIZE, type_vocab.getID(PADDING_TOKEN))

for b in range(BATCH_SIZE):
  assert not (targets1[b] == targets2[b]).all(), f"Targets should be different, instead they overlap (batch {b}): {targets1[b]}, {targets2[b]}"

print("TEST 2: PASSED")

# --------------------- VISUAL TEST ---------------------
print("TEST 3: Visual test")
ARTICLES = 500

data = load_data("data/dataset/train.json", device, INPUT_SIZE, OUTPUT_SIZE, ARTICLES, type_vocab, value_vocab, token_vocab)

index = randint(0, len(data)-1)

print("Example: ", index)

print(type_vocab.tensorToString(data[index][0]))
print(value_vocab.tensorToString(data[index][1]))
print(token_vocab.tensorToString(data[index][3]))

print("TEST 3: PASSED?")