import os
import ast
import json
import spacy

path = 'dutch_model_correct_predictions.txt'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read().splitlines()
dutch_correct = [line for line in content if line != '']

path = 'dutch_model_wrong_predictions.txt'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read().splitlines()
dutch_wrong = [line for line in content if line != '']

path = 'restaurant_all_correct_predictions.txt'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read().splitlines()
all_correct = [line for line in content if line != '']

path = 'restaurant_all_wrong_predictions.txt'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read().splitlines()
all_wrong = [line for line in content if line != '']


dutch_correct_sents = {line for line in dutch_correct if line.startswith('input:')}
all_correct_sents = {line for line in all_correct if line.startswith('input:')}

dutch_better_sents = list(dutch_correct_sents - all_correct_sents)
all_better_sents = list(all_correct_sents - dutch_correct_sents)


dutch_better = []
all_worse = []
for sentence in dutch_better_sents:
    idx = dutch_correct.index(sentence)
    dutch_better.append(dutch_correct[idx:idx + 3])

    idx = all_wrong.index(sentence)
    all_worse.append(all_wrong[idx:idx + 3])

all_better = []
dutch_worse = []
for sentence in all_better_sents:
    idx = all_correct.index(sentence)
    all_better.append(all_correct[idx:idx + 3])

    idx = dutch_wrong.index(sentence)
    dutch_worse.append(dutch_wrong[idx:idx + 3])

all_worse_labels = []
all_worse_prediction = []
for sentence in all_worse:
    labels = ast.literal_eval(sentence[1].strip('labels:'))
    predictions = ast.literal_eval(sentence[2].strip('predictions:'))
    for i in range(len(labels)):
        if labels[i] != predictions[i]:
            all_worse_labels.append(labels[i])
            all_worse_prediction.append(predictions[i])

dutch_worse_labels = []
dutch_worse_prediction = []
for sentence in dutch_worse:
    labels = ast.literal_eval(sentence[1].strip('labels:'))
    predictions = ast.literal_eval(sentence[2].strip('predictions:'))
    for i in range(len(labels)):
        if labels[i] != predictions[i]:
            dutch_worse_labels.append(labels[i])
            dutch_worse_prediction.append(predictions[i])

os.system('python3 -m spacy download nl_core_news_sm')
nlp = spacy.load('nl_core_news_sm')

x = 0
