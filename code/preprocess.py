import json
import nltk
import copy
import random
import itertools
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import XLMRobertaTokenizerFast


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    sentences = [line.split('LABELS:')[0] for line in tqdm(lines)]
    sentences = [nltk.word_tokenize(sent) for sent in tqdm(sentences)]

    aspects = []
    for line in tqdm(lines):
        sentence_aspects = []
        for word in line.split('LABELS:')[1].split():
            if word.isupper() and word not in ['CATEGORY1:', 'CATEGORY2:', 'TARGET:']:
                sentence_aspects.append(word)
        aspects.append(sentence_aspects)

    targets = []
    for line in tqdm(lines):
        sentence_targets = []
        for piece in line.split('LABELS:')[1].split('TARGET')[1:]:
            piece_targets = []
            for word in piece.split():
                if not word.isupper() and word not in ['restaurant', 'camera', 'None'] and len(word) > 1:
                    piece_targets.append(word)
            if len(piece_targets) != 0:
                sentence_targets.append(' '.join(piece_targets))
        targets.append(sentence_targets)

    return sentences, aspects, targets


def remove_low_count_labels(labels, sentences):
    remove = [key for key, value in Counter(itertools.chain(*labels)).items() if value < 20]
    keep_labels = []
    keep_sents = []
    for i, label in enumerate(labels):
        if not set(label).intersection(set(remove)):
            keep_labels.append(label)
            keep_sents.append(sentences[i])
    return keep_labels, keep_sents


def resample(labels, sentences):
    scaling = 0

    no_aspect_labels = []
    no_aspect_sentences = []
    aspect_labels = []
    aspect_sentences = []
    for i, label in enumerate(labels):
        if all([token == 'O' for token in label]):
            no_aspect_labels.append(label)
            no_aspect_sentences.append(sentences[i])
        else:
            aspect_labels.append(label)
            aspect_sentences.append(sentences[i])

    pairs = list(zip(no_aspect_labels, no_aspect_sentences))
    random.shuffle(pairs)
    no_aspect_labels = [pair[0] for pair in pairs]
    no_aspect_sentences = [pair[1] for pair in pairs]
    labels = aspect_labels + no_aspect_labels[:scaling * len(aspect_labels)]
    sentences = aspect_sentences + no_aspect_sentences[:scaling * len(aspect_sentences)]

    return labels, sentences


def get_training_label_ids(data_type):
    tag2id = json.load(open(f'./{data_type}_tag2id.json', 'r', encoding='utf-8'))
    id2tag = json.load(open(f'./{data_type}_id2tag.json', 'r', encoding='utf-8'))
    return tag2id, id2tag


def generate_training_label_ids(unique_tags, data_type):
    tag2id = {tag: id for id, tag in tqdm(enumerate(unique_tags))}
    id2tag = {id: tag for tag, id in tqdm(tag2id.items())}
    json.dump(tag2id, open(f'./{data_type}_tag2id.json', 'w', encoding='utf-8'))
    json.dump(id2tag, open(f'./{data_type}_id2tag.json', 'w', encoding='utf-8'))
    return tag2id, id2tag


def generate_labels(sentences, aspects, targets, mode='train', data_type=None):
    labels = match_BIO_tags(sentences, aspects, targets)
    labels, sentences = remove_low_count_labels(labels, sentences)
    labels, sentences = resample(labels, sentences)
    unique_tags = sorted(list(set(label for doc in tqdm(labels) for label in doc)))

    if mode == 'train':
        tag2id, id2tag = generate_training_label_ids(unique_tags, data_type)
    elif mode == 'test':
        tag2id, id2tag = get_training_label_ids(data_type)

    return labels, sentences, tag2id, id2tag


def encode_data(sentences, labels, tag2id, id2tag):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
    encodings = tokenizer(sentences, is_split_into_words=True, return_offsets_mapping=True, padding=True)
    labels = encode_tags(labels, encodings, tag2id, id2tag)
    return encodings, labels


def single_token_target(BIO_sent_tags, sents, aspects, targets, token, sent_i, token_i):
    target_bools = [token.lower() == word.lower() for target in targets[sent_i] for word in target.split()]
    if any(target_bools):
        aspect = aspects[sent_i][[i for i, target_bool in enumerate(target_bools) if target_bool][0]]
        if token_i == 0 or BIO_sent_tags[token_i - 1] == 'O':
            BIO_sent_tags[token_i] = 'B-' + aspect
        else:
            BIO_sent_tags[token_i] = 'I-' + aspect

    return BIO_sent_tags


def multiple_token_target(BIO_sent_tags, sents, aspects, targets, token, sent_i, token_i):
    target_bools = []
    for target in targets[sent_i]:
        target_bools.append([token == word for word in target.split()])

    target_bools = [any(target_bool) for target_bool in target_bools]
    if any(target_bools):
        aspect = aspects[sent_i][[i for i, target_bool in enumerate(target_bools) if target_bool][0]]
        if token_i == 0 or BIO_sent_tags[token_i - 1] == 'O':
            BIO_sent_tags[token_i] = 'B-' + aspect
        else:
            BIO_sent_tags[token_i] = 'I-' + aspect

    return BIO_sent_tags

    pass


def match_BIO_tags(sents, aspects, targets):
    aspects = [['None'] if len(aspect) == 0 else aspect for aspect in aspects]
    targets = [['None'] if len(target) == 0 else target for target in targets]

    BIO_tags = []
    for sent_i, sent in enumerate(sents):
        BIO_sent_tags = ['O'] * len(sent)
        for token_i, token in enumerate(sent):
            if any([len(target.split()) > 1 for target in targets[sent_i]]):
                BIO_sent_tags = multiple_token_target(BIO_sent_tags, sents, aspects, targets, token, sent_i, token_i)
            else:
                BIO_sent_tags = single_token_target(BIO_sent_tags, sents, aspects, targets, token, sent_i, token_i)
        BIO_tags.append(BIO_sent_tags)
    
    return BIO_tags


def fix_encodings(encodings):
    encs = copy.deepcopy(encodings)
    encs['input_ids'] = []
    encs['attention_mask'] = []
    encs['offset_mapping'] = []
    
    for i, encoding in tqdm(enumerate(encodings['input_ids'])):

        empty_strings = 0
        input_ids = []
        attention_mask = []
        offset_mapping = []
        for j, item in enumerate(encoding):
            if item != 6:
                input_ids.append(encodings['input_ids'][i][j])
                attention_mask.append(encodings['attention_mask'][i][j])
                offset_mapping.append(encodings['offset_mapping'][i][j])

            else:
                empty_strings += 1

        encs['input_ids'].append(input_ids + ([1] * empty_strings))
        encs['attention_mask'].append(attention_mask + ([0] * empty_strings))
        encs['offset_mapping'].append(offset_mapping + ([(0, 0)] * empty_strings))
        
    return encs


def encode_tags(tags, encodings, tag2id, id2tag):
    encodings = fix_encodings(encodings)
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    i = 0
    for doc_labels, doc_offset in tqdm(zip(labels, encodings.offset_mapping)):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
        i += 1

    return encoded_labels