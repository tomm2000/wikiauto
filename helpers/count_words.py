import json
from helpers.helpers import readLines, print_progress

LIMIT = 1000000
iter = 0
fails = 0

type_count = {}
value_count = {}
token_count = {}

data_file = 'data/clean/combined_data_train.json'
data_lines = readLines(data_file, LIMIT)

for line in data_lines:
  if iter >= LIMIT:
    break
  iter += 1
  print_progress(min(LIMIT, len(data_lines)), iter, 'counting', 10000)

  try:
    json_line = json.loads("{" + line + "}")
  except:
    fails += 1
    continue

  for article_name in json_line: # the article title, only 1 key
    for type in json_line[article_name]["types"]:
      if type not in type_count:
        type_count[type] = 0
      type_count[type] += 1
    
    for value in json_line[article_name]["values"]:
      if value not in value_count:
        value_count[value] = 0
      value_count[value] += 1
    
    for token in json_line[article_name]["tokens"]:
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
