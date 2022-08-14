import re
import unicodedata
import json

def print_progress(total, iter, name, every=5000):
  if(iter % every == 0 or iter == total):
    print(f"{name}: {iter}/{total} ({round(iter/total*100)}%)")

# Read a file and split into lines
def readLines(filename, size=10):
  with open(filename, 'r', encoding='utf-8') as some_file:
    l = []
    i = 0
    for line in some_file:
      # line.decode('utf-8')
      # line = line.encode('utf-8').decode()
      l.append(line.lower().replace('@@ ', ''))
      i += 1
      if i >= size and size > 0:
        break
    return l