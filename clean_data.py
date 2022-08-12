import json
from helpers import readLines, print_progress

LIMIT = 1000000
iter = 0

# ----- ARTICLE -----
iter = 0
article_file = 'data/raw/train-500k.json'
article_lines = readLines(article_file, LIMIT)
article_data = {} # article: [string]

for line in article_lines:
  if iter >= LIMIT:
    break
  iter += 1
  print_progress(min(LIMIT, len(article_lines)), iter, 'article')

  json_line = json.loads(line)

  article_data[json_line["doc_title"]] = json_line["text"].split(" ")

del article_lines
# ----- TABLE -----
table_data = {} # article: { types: [string], values: [string] }

#/ --- wikidata ---
iter = 0
wikidata_file = 'data/raw/wikidata.json'
wikidata_lines = readLines(wikidata_file, LIMIT)

for line in wikidata_lines:
  if iter >= LIMIT:
    break
  iter += 1
  print_progress(min(LIMIT, len(wikidata_lines)), iter, 'wikidata')

  json_line = json.loads(line)

  article_name = json_line["wikidata_name"]

  if article_name not in article_data:
    continue

  clean_line = { "types": [], "values": [] }
  if article_name in table_data:
    clean_line = table_data[article_name]

  for key in json_line["wikidata_details"]:
    for value in json_line["wikidata_details"][key]:
      clean_line["types"].append(key)
      clean_line["values"].append(value["data"])

  table_data[article_name] = clean_line

del wikidata_lines
#/ --- infobox ---
iter = 0
infobox_file = 'data/raw/infobox.json'
infobox_lines = readLines(infobox_file, LIMIT)

for line in infobox_lines:
  if iter >= LIMIT:
    break
  iter += 1
  print_progress(min(LIMIT, len(infobox_lines)), iter, 'infobox')

  json_line = json.loads(line)

  article_name = json_line["title"]
  if article_name not in article_data:
    continue

  clean_line = { "types": [], "values": [] }
  if article_name in table_data:
    clean_line = table_data[article_name]

  for key in json_line["infobox"]:
    clean_line["types"].append(key)
    clean_line["values"].append(json_line["infobox"][key])

del infobox_lines
# ----- COMBINED -----

# clears the output file
with open('data/clean/combined_data.json','w') as f:
  pass

iter = 0

for article_name in article_data:
  if article_name in table_data:
    combined_data = {
      "types": table_data[article_name]["types"],
      "values": table_data[article_name]["values"],
      "tokens": article_data[article_name]
    }

    with open('data/clean/combined_data.json', 'a', encoding='utf-8') as outfile:
      text = json.dumps(combined_data, ensure_ascii=False)
      outfile.write(f"\"{article_name}\": {text}\n")

  article_data[article_name] = None
  table_data[article_name] = None

  print_progress(len(article_data), iter, 'combined')
  iter += 1