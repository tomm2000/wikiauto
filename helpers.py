import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def timeSince(since, percent):
  now = time.time()
  s = now - since
  es = s / (percent)
  rs = es - s
  return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
  plt.figure()
  fig, ax = plt.subplots()
  # this locator puts ticks at regular intervals
  loc = ticker.MultipleLocator(base=0.01)
  ax.yaxis.set_major_locator(loc)
  plt.plot(points)
  plt.show()