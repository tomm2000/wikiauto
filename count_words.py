import json
from lib.generic import readLines
from tqdm import tqdm

fails = 0

type_count = {}
value_count = {}
token_count = {}

data_file = 'data/clean/dataset.json'
lines = readLines(data_file, -1)

# {"Name_ID": "10th edition of Systema Naturae", "edition or translation of": [{"mainsnak": "Systema Naturae"}], "instance of": [{"mainsnak": "Edition (book)"}], "author": [{"mainsnak": "Carl Linnaeus"}], "publication date": [{"mainsnak": "1758"}], "TEXT": [["the", "'", "10th", "edition", "of", "Systema Naturae", "'", "is", "a", "book", "written", "by", "Carl Linnaeus", "and", "published", "in", "two", "volumes", "in", "1758", "and", "1759", "which", "marks", "the", "starting", "point", "of", "zoological nomenclature", "."], ["before", "1758", "most", "biological", "catalogues", "had", "used", "polynomial", "names", "for", "the", "taxa", "included", "including", "earlier", "editions", "of", "Systema Naturae", "."], ["the", "first", "work", "to", "consistently", "apply", "binomial", "nomenclature", "across", "the", "animal", "kingdom", "was", "the", "10th", "edition", "of", "Systema Naturae", "."], ["the", "International Commission on Zoological Nomenclature", "therefore", "chose", "1", "January", "1758", "as", "the", "\"", "starting", "point", "\"", "for", "zoological", "nomenclature", "and", "asserted", "that", "the", "10th", "edition", "of", "Systema Naturae", "was", "to", "be", "treated", "as", "if", "published", "on", "that", "date", "."], ["the", "only", "work", "which", "takes", "priority", "over", "the", "10th", "edition", "is", "Carl Alexander Clerck", "'s", "'", "or", "'", "which", "was", "published", "in", "1757", "but", "is", "also", "to", "be", "treated", "as", "if", "published", "on", "January", "1", "1758", "."], ["during", "Linnaeus", "'", "lifetime", "Systema Naturae", "was", "under", "continuous", "revision", "."], ["title", "page", "of", "the", "10th", "edition", "of", "Systema Naturae", "."], ["an", "oil", "painting", "of", "Carl Linnaeus", "by", "Alexander Roslin", "in", "1775", "."], ["the", "common cuttlefish", "was", "named", "Sepia", "officinalis", "in", "the", "10th", "edition", "of", "Systema Naturae", "."], ["allionia incarnata", "was", "one", "of", "the", "two", "new", "species", "in", "the", "new", "genus", "Allionia", "introduced", "in", "the", "10th", "edition", "of", "Systema Naturae", "."]]}

for i in tqdm(range(0, len(lines)), desc="Cleaning data"):
  line = lines[i]

  try:
    json_line = json.loads(line)
  except:
    fails += 1
    continue

  for type in json_line["types"]:
    if type not in type_count:
      type_count[type] = 0
    type_count[type] += 1
  
  for value in json_line["values"]:
    if value not in value_count:
      value_count[value] = 0
    value_count[value] += 1
  
  for token in json_line["tokens"]:
    if token not in token_count:
      token_count[token] = 0
    token_count[token] += 1


print("failed to load: " + str(fails))


# clears the output file
with open('data/counts/types.txt','w') as f:
  pass

with open('data/counts/values.txt','w') as f:
  pass

with open('data/counts/tokens.txt','w') as f:
  pass

with open('data/counts/types.txt', 'a', encoding='utf-8') as outfile:
  data = [(key, type_count[key]) for key in type_count]
  data.sort(key=lambda x: x[1], reverse=True)
  for t in data:
    outfile.write(f"{t[0]}\t{t[1]}\n")


with open('data/counts/values.txt', 'a', encoding='utf-8') as outfile:
  data = [(key, value_count[key]) for key in value_count]
  data.sort(key=lambda x: x[1], reverse=True)
  for t in data:
    outfile.write(f"{t[0]}\t{t[1]}\n")


with open('data/counts/tokens.txt', 'a', encoding='utf-8') as outfile:
  data = [(key, token_count[key]) for key in token_count]
  data.sort(key=lambda x: x[1], reverse=True)
  for t in data:
    outfile.write(f"{t[0]}\t{t[1]}\n")
