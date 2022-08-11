from base64 import encode
import json
from helpers import readLines, print_progress

LIMIT = 1000000
iter = 0

# ----- WIKIDATA -----
iter = 0
wikidata_file = 'data/raw/wikidata.json'
wikidata_lines = readLines(wikidata_file, LIMIT)

# clears the output file
with open('data/clean/wikidata.json','w') as f:
  pass

for line in wikidata_lines:
  if iter >= LIMIT:
    break
  iter += 1
  print_progress(min(LIMIT, len(wikidata_lines)), iter, 'wikidata')

  json_line = json.loads(line)

  clean_line = { "article": "", "data": {} }

  clean_line["article"] = json_line["wikidata_name"]

  for key in json_line["wikidata_details"]:
    value_list = []
    for value in json_line["wikidata_details"][key]:
      value_list.append(value["data"])
    clean_line["data"][key] = value_list

  with open('data/clean/wikidata.json', 'a', encoding='utf-8') as outfile:
    text = json.dumps(clean_line, ensure_ascii=False)
    outfile.write(f"{text}\n")

# ----- INFOBOX -----
iter = 0
infobox_file = 'data/raw/infobox.json'
infobox_lines = readLines(infobox_file, LIMIT)

# clears the output file
with open('data/clean/infobox.json','w') as f:
  pass

for line in infobox_lines:
  if iter >= LIMIT:
    break
  iter += 1
  print_progress(min(LIMIT, len(infobox_lines)), iter, 'infobox')

  json_line = json.loads(line)

  clean_line = { "article": "", "data": {} }

  clean_line["article"] = json_line["title"]

  for key in json_line["infobox"]:
    clean_line["data"][key] = json_line["infobox"][key]

  with open('data/clean/infobox.json', 'a', encoding='utf-8') as outfile:
    text = json.dumps(clean_line, ensure_ascii=False)
    outfile.write(f"{text}\n")

# ----- ARTICLE -----
iter = 0
article_file = 'data/raw/train-500k.json'
article_lines = readLines(article_file, LIMIT)

# clears the output file
with open('data/clean/article.json','w') as f:
  pass

for line in article_lines:
  if iter >= LIMIT:
    break
  iter += 1
  print_progress(min(LIMIT, len(article_lines)), iter, 'article')

  json_line = json.loads(line)

  clean_line = { "article": "", "data": {} }

  clean_line["article"] = json_line["doc_title"]

  clean_line["data"] = json_line["text"].split(" ")

  with open('data/clean/article.json', 'a', encoding='utf-8') as outfile:
    text = json.dumps(clean_line, ensure_ascii=False)
    outfile.write(f"{text}\n")