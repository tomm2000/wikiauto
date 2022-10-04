# 1: installare librerie
- pytorch
- colorama (colori nel terminale)
- tqdm (barra di caricamento)

# 2: unzippare il file `data.zip` in una cartella `/data`
dovrebbe essere:\
|-`./data`\
|---`/clean`\
|----- `dataset.json`\
|---`/counts`\
|----- `tokens.txt`\
|----- `types.txt`\
|----- `values.txt`

# 3: il file `setup/main/setup.json` contiene il setup per il modello
- in "model_path" mettere il path in salvare il modello (e da dove verrà caricato)
- in "result_path" mettere il path dove verranno salvati i dati di output

# 4: il file `setup/main/data.json` contiene i dati salvati del modello (losses e perplexity)

# 5: eseguire il file `train.py`