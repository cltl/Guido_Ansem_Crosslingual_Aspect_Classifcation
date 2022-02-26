import glob
import random
import itertools
import numpy as np
from collections import Counter
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt


def get_path(dataset:str='Restaurant', language: str='English', mode: str='Train'):
    file_type = {'Test': 'B', 
                    'Train':'xml'}

    languages = {'Restaurant':{'Dutch':'Dutch',
                                'English':'English',
                                'French':'French',
                                'Russian':'Russian',
                                'Spanish':'Spanish',
                                'Turkish':'Turkish',
                                'All':'*'},

                    'camera':{'German':'de',
                                'English':'en',
                                'Spanish':'es',
                                'French':'fr',
                                'Italian':'it',
                                'All': '*'}
                                }

    return f'./data/{mode}/raw/{dataset}*_{languages[dataset][language]}.{file_type[mode]}'


def get_data(dataset:str='Restaurant', language: str='English', mode: str='Train') -> list:
    folder_path = get_path(dataset, language, mode)

    data = []
    skipped = 0
    multi_token = 0
    single_token = 0
    token_lengths = []
    for path in glob.glob(folder_path):
        with open(path, 'r', encoding='utf-8') as file:
            tree = ET.parse(path)
        reviews = tree.getroot()
        for review in reviews:
            datum = []
            for sentence in review[0]:
                opinions = []
                for i, item in enumerate(sentence):
                    if item.tag.lower() == 'text':
                        if item.text != None:
                            text = item.text
                        elif 'text' in item.attrib.keys():
                            text = item.attrib['text']
                        else:
                            skipped += 1

                        if len(sentence) > 1:
                            item = sentence[i + 1]
                            for opinion in item:
                                opinions.append(opinion.attrib)
                        else:
                            opinions.append({"category":None, "target":None})

                if len(opinion.attrib['target'].split()) > 1:
                    multi_token += 1
                elif len(opinion.attrib['target'].split()) == 1:
                    single_token += 1
                token_lengths.append(len(opinion.attrib['target'].split()))

                if len(opinions) != 0:
                    if 'camera' in path.split("/")[-1]:
                        main_cat =  'camera'
                    else:
                        main_cat = 'restaurant'
                    labels = 'LABELS: ' + ' '.join([' '.join([f'CATEGORY1: {main_cat}', 
                                                    f'CATEGORY2: {opinion["category"]}', 
                                                    f'TARGET: {opinion["target"]}']) 
                                                    for opinion in opinions])

                    data.append((text, labels))
    return data


def show_plot(data:list, dataset: str='Restaurant', language='English', mode='Train'):
    labels = [datum[1].split()[2::2] for datum in data]
    labels = [label for label in list(itertools.chain(*labels)) if label.isupper()]
    labels = Counter(labels)

    if dataset == 'camera':
        labels = {key:value for key, value in labels.items() if key in camera_labels}
    elif dataset == 'Restaurant':
        labels = {key:value for key, value in labels.items() if key in restaurant_labels}

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = {k:v for k, v in sorted(labels.items())}
    keys = values = ['#\n'.join(key.split('#')) for key in list(labels.keys())]
    values = list(labels.values())
    y_pos = np.arange(len(keys))

    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()
    ax.set_xlabel('Count')

    for i, v in enumerate(labels.values()):
        ax.text(x=i-.35, y=v+1, s=str(v), color='black', fontsize=12)

    title = f'{dataset} {mode} {language}'
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f'Figures/{dataset}_{language}_{mode}_Labels.png', format='png')


def plot_label_distribution(dataset:str='Restaurant', language: str='English', mode: str='Train'):
    data = get_data(dataset, language, mode)
    show_plot(data, dataset, language, mode)
    print('Done!')


camera_labels = ['IMAGING', 'GENERAL', 'OTHER', 'PRICE', 'DISPLAY', 'EXPOSURE', 'DIMENSION', 'LENS', 'BATTERY', 'VIDEO', 'ZOOM', 'PERFORMANCE', 'MEMORY']
restaurant_labels = ['RESTAURANT#GENERAL', 'NULL', 'SERVICE#GENERAL', 'AMBIENCE#GENERAL', 'FOOD#QUALITY', 'FOOD#PRICES', 'RESTAURANT#PRICES', 'FOOD#STYLE_OPTIONS', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS', 'LOCATION#GENERAL', 'DRINKS#PRICES', 'RESTAURANT#MISCELLANEOUS']

plot_label_distribution(dataset='Restaurant', language='Dutch', mode='Train')
