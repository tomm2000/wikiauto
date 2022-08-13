from ctypes import sizeof
import json
from helpers import readLines, print_progress

LIMIT = 1000000
iter = 0

# ----- TABLE -----
table_data = {} # article: { types: [string], values: [string] }

#/ --- wikidata ---
iter = 0
wikidata_file = 'data/raw/wikidata.json'
wikidata_lines = readLines(wikidata_file, LIMIT)

for i in range(min(LIMIT, len(wikidata_lines))):
# for line in wikidata_lines:
  line = wikidata_lines[i]
  print_progress(min(LIMIT, len(wikidata_lines)), i, 'wikidata')

  json_line = json.loads(line)

  article_name = json_line["wikidata_name"]

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
with open('data/clean/combined_data_eval.json','w') as f:
  pass

iter = 0

with open('data/clean/combined_data_eval.json', 'a', encoding='utf-8') as outfile:
  for article_name in table_data:
    combined_data = {
      "types": table_data[article_name]["types"],
      "values": table_data[article_name]["values"],
    }

    text = json.dumps(combined_data, ensure_ascii=False)
    outfile.write(f"\"{article_name}\": {text}\n")

    table_data[article_name] = None

    print_progress(len(table_data), iter, 'combined')
    iter += 1