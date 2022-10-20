from lib.generic import readLines

UNKNOWN_TOKEN = '<UNK>'
PADDING_TOKEN = '<PAD>'
START_TOKEN = '<SOS>'
END_TOKEN = '<EOS>'

class vocab:
  def __init__(self, max_size = 50000):
    self.word2idx = { UNKNOWN_TOKEN: 0, PADDING_TOKEN: 1 }
    self.idx2word = { 0: UNKNOWN_TOKEN, 1: PADDING_TOKEN }
    self.max_size = max_size

  def addWord(self, word):
    if len(self) >= self.max_size:
      return
    if word not in self.word2idx:
      self.word2idx[word] = len(self)
      self.idx2word[len(self)] = word

  def load_file(self, filename):
    with open(filename, 'r', encoding='utf-8') as f:
      lines = readLines(filename, size=-1)
      for line in lines:
        word = line.split('\t')[0]
        self.addWord(word)
  
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
      raise KeyError(f'Idx [{idx}] not found in vocab')

  def tensorToString(self, tensor, add_id = False):
    sentence = ""

    for t in range(tensor.size(0)):
      if add_id:
        sentence += f"({tensor[t].item()}) "
      sentence += self.getWord(tensor[t].item()) + " | "

    return sentence