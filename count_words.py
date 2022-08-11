from base64 import encode
import json
from helpers import readLines, print_progress

LIMIT = 1000000
iter = 0

type_count = {}
value_count = {}
token_count = {}

# ----- WIKIDATA -----
wikidata_file = 'data/clean/wikidata.json'
wikidata_lines = readLines(wikidata_file, LIMIT)

for line in wikidata_lines:
  if iter >= LIMIT:
    break
  iter += 1
  print_progress(min(LIMIT, len(wikidata_lines)), iter, 'wikidata', 10000)

  json_line = json.loads(line)

  for key in json_line["data"]:
    if key not in type_count:
      type_count[key] = 0
    type_count[key] += 1

    for value in json_line["data"][key]:
      if value not in value_count:
        value_count[value] = 0
      value_count[value] += 1

# ----- INFOBOX -----
# iter = 0
# infobox_file = 'data/clean/infobox.json'
# infobox_lines = readLines(infobox_file, LIMIT)

# for line in infobox_lines:
#   if iter >= LIMIT:
#     break
#   iter += 1
#   print_progress(min(LIMIT, len(infobox_lines)), iter, 'infobox', 10000)

#   json_line = json.loads(line)

#   for key in json_line["data"]:
#     if key not in type_count:
#       type_count[key] = 0
#     type_count[key] += 1

#     value = json_line["data"][key]

#     if value not in value_count:
#       value_count[value] = 0
#     value_count[value] += 1

# ----- VALUE & TYPE FILE -----

# clears the output file
with open('data/counts/types.txt','w') as f:
  pass

with open('data/counts/values.txt','w') as f:
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

# ----- ARTICLE -----
iter = 0
article_file = 'data/clean/article.json'
article_lines = readLines(article_file, LIMIT)

for line in article_lines:
  if iter >= LIMIT:
    break
  iter += 1
  print_progress(min(LIMIT, len(article_lines)), iter, 'article', 10000)

  json_line = json.loads(line)

  for token in json_line["data"]:
    # print(token)
    if token not in token_count:
      token_count[token] = 0
    token_count[token] += 1

# clears the output file
with open('data/counts/tokens.txt','w') as f:
  pass

with open('data/counts/tokens.txt', 'a', encoding='utf-8') as outfile:
  data = [(key, token_count[key]) for key in token_count]
  data.sort(key=lambda x: x[1], reverse=True)
  for t in data:
  # text = json.dumps(token_count, ensure_ascii=False)
  # outfile.write(f"{text}\n")
    outfile.write(f"{t[0]}\t{t[1]}\n")