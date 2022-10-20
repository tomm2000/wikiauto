import json
from lib.generic import readLines
from tqdm import tqdm


def clean_file(lines, output_file):
  with open(output_file, 'w', encoding='utf-8') as _: pass

  with open(output_file, 'a', encoding='utf-8') as outfile:
    for line in tqdm(lines, desc="Cleaning data"):
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

lines = readLines('data/dataset/raw.json')

train_split = 0.85
train_lines = lines[:int(len(lines) * train_split)]

valid_split = 0.10
valid_lines = lines[int(len(lines) * train_split):int(len(lines) * (train_split + valid_split))]

test_lines = lines[int(len(lines) * (train_split + valid_split)):]

clean_file(train_lines, 'data/dataset/train.json')
clean_file(valid_lines, 'data/dataset/validation.json')
clean_file(test_lines, 'data/dataset/test.json')