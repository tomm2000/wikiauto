import matplotlib.pyplot as plt

def readLines(filename, size=-1):
  '''
  reads a file and returns a list of lines
  with size < 0, all lines are returned
  '''
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

def plotResults(train_losses, valid_losses, train_perplexities, valid_perplexities):
  plt.figure()
  fig, ax = plt.subplots(nrows=2, ncols=1)

  ax[0].plot(train_losses)
  ax[0].plot(valid_losses)
  ax[0].axhline(y=0, color='k')
  ax[0].grid(True, which='both')
  ax[0].legend(['train', 'valid'])
  
  ax[1].plot(train_perplexities)
  ax[1].plot(valid_perplexities)
  ax[1].axhline(y=0, color='k')
  ax[1].grid(True, which='both')
  ax[1].legend(['train', 'valid'])

  return fig
