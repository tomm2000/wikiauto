from cmath import isnan
import json
import time
import math
import matplotlib.pyplot as plt

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


def loadTrainData(path):
  return json.load(open(path, "r"))
  
def saveTrainData(file, epoch, iter_losses, train_losses, eval_losses, train_perplexity, eval_perplexity, prec_loss, flat):
  data = loadTrainData(file)

  data["epoch"] = epoch
  data["iter_losses"] = iter_losses
  data["train_losses"] = train_losses
  data["eval_losses"] = eval_losses
  data["train_perplexity"] = train_perplexity
  data["eval_perplexity"] = eval_perplexity
  data["prec_loss"] = prec_loss
  data["flat"] = flat

  with open(file, "w") as f:
    f.write(json.dumps(data))


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

def asMsecs(s):
  sec = math.floor(s)
  msec = s * 1000
  msec -= sec * 1000
  return '%ds %dms' % (sec, msec)


def getPlot(plots):
  plt.figure()
  fig, ax = plt.subplots(nrows=len(plots), ncols=1)

  for i in range(len(plots)):
    for j in range(len(plots[i])):
      ax[i].plot(plots[i][j])
      ax[i].axhline(y=0, color='k')
      ax[i].grid(True, which='both')
    
  return fig

def calc_avg_loss(prec_loss, curr_loss, alpha=0.95):
  if prec_loss == 0:
    return curr_loss
  return alpha * prec_loss + (1.0 - alpha) * curr_loss


class timelog:
  def __init__(self, name):
    self.name = name
    self.start = time.time()

  def log(self, msg, pre = "", post = ""):
    print(f"{pre}{self.name}: {msg} ({asMsecs(time.time() - self.start)}){post}")

  def log_end(self, msg, pre = "", post = ""):
    self.log(msg, pre, post)
    self.start = time.time()

class stackTimelog:
  def __init__(self, name):
    self.name = name
    self.start = time.time()
    self.text = f"{self.name}: "

  def log(self, msg = ""):
    self.text += f" {msg} ({asMsecs(time.time() - self.start)}) |"

  def log_end(self, msg = ""):
    self.log(msg)
    self.start = time.time()

  def flush(self):
    print(self.text)