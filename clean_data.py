import json
from lib.generic import readLines
from tqdm import tqdm

data_file = 'data/raw/dataset.json'
lines = readLines(data_file, -1)

# {"Name_ID": "10th edition of Systema Naturae", "edition or translation of": [{"mainsnak": "Systema Naturae"}], "instance of": [{"mainsnak": "Edition (book)"}], "author": [{"mainsnak": "Carl Linnaeus"}], "publication date": [{"mainsnak": "1758"}], "TEXT": [["the", "'", "10th", "edition", "of", "Systema Naturae", "'", "is", "a", "book", "written", "by", "Carl Linnaeus", "and", "published", "in", "two", "volumes", "in", "1758", "and", "1759", "which", "marks", "the", "starting", "point", "of", "zoological nomenclature", "."], ["before", "1758", "most", "biological", "catalogues", "had", "used", "polynomial", "names", "for", "the", "taxa", "included", "including", "earlier", "editions", "of", "Systema Naturae", "."], ["the", "first", "work", "to", "consistently", "apply", "binomial", "nomenclature", "across", "the", "animal", "kingdom", "was", "the", "10th", "edition", "of", "Systema Naturae", "."], ["the", "International Commission on Zoological Nomenclature", "therefore", "chose", "1", "January", "1758", "as", "the", "\"", "starting", "point", "\"", "for", "zoological", "nomenclature", "and", "asserted", "that", "the", "10th", "edition", "of", "Systema Naturae", "was", "to", "be", "treated", "as", "if", "published", "on", "that", "date", "."], ["the", "only", "work", "which", "takes", "priority", "over", "the", "10th", "edition", "is", "Carl Alexander Clerck", "'s", "'", "or", "'", "which", "was", "published", "in", "1757", "but", "is", "also", "to", "be", "treated", "as", "if", "published", "on", "January", "1", "1758", "."], ["during", "Linnaeus", "'", "lifetime", "Systema Naturae", "was", "under", "continuous", "revision", "."], ["title", "page", "of", "the", "10th", "edition", "of", "Systema Naturae", "."], ["an", "oil", "painting", "of", "Carl Linnaeus", "by", "Alexander Roslin", "in", "1775", "."], ["the", "common cuttlefish", "was", "named", "Sepia", "officinalis", "in", "the", "10th", "edition", "of", "Systema Naturae", "."], ["allionia incarnata", "was", "one", "of", "the", "two", "new", "species", "in", "the", "new", "genus", "Allionia", "introduced", "in", "the", "10th", "edition", "of", "Systema Naturae", "."]]}

with open('data/clean/dataset.json', 'w', encoding='utf-8') as outfile:
  pass

with open('data/clean/dataset.json', 'a', encoding='utf-8') as outfile:
  for i in tqdm(range(0, len(lines)), desc="Cleaning data"):
    line = lines[i]
    json_line = json.loads(line)

    # TARGETS
    targets = json_line['text']

    for target in targets:
      article = {
        'title': json_line['name_id'],
        'tokens': target,
        'types': [],
        'values': []
      }

      for key in json_line:
        if key == 'name_id': continue
        if key == 'text': continue

        values = json_line[key]

        for value in values:
          article['types'].append(key)
          article['values'].append(value['mainsnak'])

      text = json.dumps(article, ensure_ascii=False)
      outfile.write(f"{text}\n")