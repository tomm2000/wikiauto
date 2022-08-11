import random
import json
from helpers import readLines

UNKNOWN_TOKEN = '<UNK>'
PADDING_TOKEN = '<PAD>'
START_TOKEN = '<SOS>'
END_TOKEN = '<EOS>'

class vocab:
  def __init__(self, filename, size = 50000):
    self.word2idx = { UNKNOWN_TOKEN: 0, PADDING_TOKEN: 1, START_TOKEN: 2, END_TOKEN: 3 }
    self.idx2word = { 0: UNKNOWN_TOKEN, 1: PADDING_TOKEN, 2: START_TOKEN, 3: END_TOKEN }

    with open(filename, 'r', encoding='utf-8') as f:
      lines = readLines(filename, size=-1)
      for line in lines:
        if len(self) >= size: break
        word = line.split('\t')[0]

        if word not in self.word2idx:
          self.word2idx[word] = len(self.word2idx)
          self.idx2word[len(self.idx2word)] = word
  
  def __len__(self):
    return len(self.idx2word)

  def getID(self, word):
    if word in self.word2idx:
      return self.word2idx[word]
    else:
      return self.word2idx[UNKNOWN_TOKEN]
    
  def getWord(self, idx):
    if idx in self.idx2word:
      return self.idx2word[idx]
    else:
      raise KeyError('Idx not found in vocab')