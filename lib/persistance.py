import json

def load_setup(file):
  setup = json.load(open(file, "r"))
  return setup

def load_results(file):
  try:
    data = json.load(open(file, "r"))
    return data
  except:
    return {
      "epoch": 0,
      "train_losses": [],
      "valid_losses": [],
      "train_perplexities": [],
      "valid_perplexities": [],
    }

def save_results(file, data):
  with open(file, "w") as f:
    f.write(json.dumps(data))

def save_sample(file, sample, epoch, vocab, batch=0):
  predict = vocab.tensorToString(sample[0][batch])
  target  = vocab.tensorToString(sample[1][batch])

  with open(file, "a", encoding='UTF-8') as f:
    f.write(f"------------------ Epoch {epoch} ------------------\n")
    f.write(f"Predict: {predict}\n")
    f.write(f"------------------ ↑ predict | target ↓ ------------------\n")
    f.write(f"Target: {target}\n")
    f.write(f"------------------------------------------------------------\n")