# Datensatz
Der zur Analyse verwendete Datensatz ist kostenfrei auf der Plattform kaggle zugänglich:

https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews

# Verwendung
## Voraussetzungen
Für die Ausführung des Projekts wird Python 3.x benötigt. Die erforderlichen Bibliotheken können über die bereitgestellte `requirements.txt` installiert werden.

`pip install -r requirements.txt`

## Datensatz
Der Datensatz `amazon_alexa.tsv` muss im selben Verzeichnis wie die Python-Skripte abgelegt werden.

## Ausführung
Für die Analyse stehen zwei Skripte zur Verfügung:
- `positiv_recessions.py` zur Analyse positiver Bewertungen (4-5 Sterne)
- `negativ_recessions.py` zur Analyse negativer Bewertungen (1-2 Sterne)

Die Skripte können über die Kommandozeile oder innerhalb einer Jupyter-Notebook-Umgebung ausgeführt werden:

`python positiv_recessions.py`

bzw.

`python negativ_recessions.py`

# Ergebnisse
## Auswertung Coherence Score von den negativen Bewertungen:
<img width="1230" height="873" alt="image" src="https://github.com/user-attachments/assets/2aedcd2d-ba9a-423e-a213-92cc4c9af5b7" />

## Ausgabe der Analyse von den negativen Bewertungen:
LDA:

- Thema 1
['get', 'working', 'one', 'would', 'product', 'echo', 'time', 'amazon', 'alexa', 'work']
- Thema 2
['dont', 'get', 'screen', 'amazon', 'like', 'would', 'thing', 'dot', 'device', 'echo']

LSA:

- Thema 1
['amazon', 'alexa', 'sound', 'one', 'would', 'time', 'dot', 'device', 'work', 'echo']
- Thema 2
['worked', 'dot', 'month', 'command', 'used', 'twice', 'stopped', 'didnt', 'work', 'working']


## Auswertung Coherence Score von den positiven Bewertungen:
<img width="1230" height="872" alt="image" src="https://github.com/user-attachments/assets/f2d7210f-e310-42d6-b571-e52cf9559173" />

## Ausgabe der Analyse von den positiven Bewertungen:
LDA:

- Thema 1
['perfect', 'amazing', 'quality', 'speaker', 'device', 'product', 'sound', 'good', 'work', 'great']
- Thema 2
['product', 'smart', 'alexa', 'echo', 'like', 'music', 'setup', 'use', 'set', 'easy']
- Thema 3
['product', 'everything', 'like', 'work', 'use', 'echo', 'music', 'learning', 'still', 'great']
- Thema 4
['get', 'plus', 'great', 'dot', 'sound', 'like', 'music', 'alexa', 'love', 'echo']
- Thema 5
['time', 'home', 'speaker', 'smart', 'work', 'echo', 'great', 'one', 'like', 'alexa']
- Thema 6
['like', 'dot', 'music', 'bought', 'new', 'work', 'great', 'one', 'echo', 'love']
- Thema 7
['im', 'use', 'much', 'watch', 'stick', 'fire', 'thing', 'alexa', 'tv', 'love']

LSA:

- Thema 1
['sound', 'dot', 'product', 'use', 'easy', 'alexa', 'work', 'echo', 'great', 'love']
- Thema 2
['echo', 'good', 'like', 'set', 'use', 'sound', 'easy', 'product', 'work', 'great']
- Thema 3
['alexa', 'really', 'music', 'dot', 'setup', 'echo', 'like', 'use', 'set', 'easy']
- Thema 4
['useamazing', 'affordable', 'install', 'setup', 'use', 'love', 'product', 'great', 'set', 'easy']
- Thema 5
['charm', 'love', 'new', 'perfect', 'perfectly', 'easy', 'like', 'set', 'well', 'work']
- Thema 6
['learning', 'charm', 'use', 'alexa', 'one', 'really', 'everything', 'new', 'product', 'like']
- Thema 7
['pretty', 'well', 'love', 'speaker', 'far', 'quality', 'sound', 'product', 'like', 'good']

