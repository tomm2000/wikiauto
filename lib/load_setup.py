import json

def load_setup(path):
  setup = json.load(open(f"{path}/setup.json", "r"))
  data = json.load(open(f"{path}/data.json", "r"))
  return setup, data

def save_setup(path, epoch, iter_losses, train_losses, eval_losses, train_perplexity, eval_perplexity, prec_loss, flat):
  setup, data = load_setup(path)

  setup["epoch"] = epoch
  setup["prec_loss"] = prec_loss
  setup["flat"] = flat
  
  data["iter_losses"] = iter_losses
  data["train_losses"] = train_losses
  data["eval_losses"] = eval_losses
  data["train_perplexities"] = train_perplexity
  data["eval_perplexities"] = eval_perplexity

  with open(f"{path}/data.json", "w") as f:
    f.write(json.dumps(data))

  with open(f"{path}/setup.json", "w") as f:
    f.write(json.dumps(setup))