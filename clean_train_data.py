import json
from helpers import readLines, print_progress

LIMIT = 1000000

# ----- ARTICLE -----
article_file = 'data/raw/train-500k.json'
article_lines = readLines(article_file, LIMIT)
article_data = {} # article: [string]

for i in range(min(len(article_lines), LIMIT)):
# for line in article_lines:
  line = article_lines[i]

  print_progress(min(LIMIT, len(article_lines)), i, 'article')

  json_line = json.loads(line)

  article_data[json_line["doc_title"]] = json_line["text"].split(" ")

  article_lines[i] = None

article_lines = None

# ----- TABLE -----
table_data = {} # article: { types: [string], values: [string] }

#/ --- wikidata ---
wikidata_file = 'data/raw/wikidata.json'
wikidata_lines = readLines(wikidata_file, LIMIT)

for i in range(min(len(wikidata_lines), LIMIT)):
# for line in wikidata_lines:
  line = wikidata_lines[i]
  print_progress(min(LIMIT, len(wikidata_lines)), i, 'wikidata')

  json_line = json.loads(line)

  article_name = json_line["wikidata_name"]

  if article_name not in article_data:
    wikidata_lines[i] = None
    continue

  clean_line = { "types": [], "values": [] }
  if article_name in table_data:
    clean_line = table_data[article_name]

  for key in json_line["wikidata_details"]:
    for value in json_line["wikidata_details"][key]:
      clean_line["types"].append(key)
      clean_line["values"].append(value["data"])

  table_data[article_name] = clean_line

  # removes elements that are not needed anymore to save memory
  wikidata_lines[i] = None

wikidata_lines = None

#/ --- infobox ---
infobox_file = 'data/raw/infobox.json'
infobox_lines = readLines(infobox_file, LIMIT)

for i in range(min(len(infobox_lines), LIMIT)):
# for line in infobox_lines:
  line = infobox_lines[i]
  print_progress(min(LIMIT, len(infobox_lines)), i, 'infobox')

  json_line = json.loads(line)

  article_name = json_line["title"]
  if article_name not in article_data:
    infobox_lines[i] = None
    continue

  clean_line = { "types": [], "values": [] }
  if article_name in table_data:
    clean_line = table_data[article_name]

  for key in json_line["infobox"]:
    clean_line["types"].append(key)
    clean_line["values"].append(json_line["infobox"][key])
  
  # removes elements that are not needed anymore to save memory
  infobox_lines[i] = None

infobox_lines = None


# ----- COMBINED -----

# clears the output file
with open('data/clean/combined_data_train.json','w') as f:
  pass

iter = 0

with open('data/clean/combined_data_train.json', 'a', encoding='utf-8') as outfile:
  for article_name in article_data:
    if article_name in table_data:
      combined_data = {
        "types": table_data[article_name]["types"],
        "values": table_data[article_name]["values"],
        "tokens": article_data[article_name]
      }

      text = json.dumps(combined_data, ensure_ascii=False)
      outfile.write(f"\"{article_name}\": {text}\n")

    article_data[article_name] = None
    table_data[article_name] = None

    print_progress(len(article_data), iter, 'combined')
    iter += 1