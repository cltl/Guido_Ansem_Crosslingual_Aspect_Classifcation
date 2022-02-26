import os
import glob
import json
import spacy
from tqdm import tqdm
from collections import Counter
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt

camera_labels = ['IMAGING', 'GENERAL', 'OTHER', 'PRICE', 'DISPLAY', 'EXPOSURE', 'DIMENSION', 'LENS', 'BATTERY', 'VIDEO', 'ZOOM', 'PERFORMANCE', 'MEMORY']
restaurant_labels = ['RESTAURANT#GENERAL', 'NULL', 'SERVICE#GENERAL', 'AMBIENCE#GENERAL', 'FOOD#QUALITY', 'FOOD#PRICES', 'RESTAURANT#PRICES', 'FOOD#STYLE_OPTIONS', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS', 'LOCATION#GENERAL', 'DRINKS#PRICES', 'RESTAURANT#MISCELLANEOUS']


def get_file_name(language: str='English') -> str:
    file_dict = {'English': 'cam*en.xml',
                'Dutch': 'Res*Du*.xml',
                'German': 'cam*de.xml',
                'French': 'cam*fr.xml',
                'Italian': 'cam*it.xml',
                'Spanish': 'cam*es.xml', 
                'Russian': 'Res*Rus*.xml'}
    return file_dict[language]


def get_model_name(language: str='English') -> str:
    model_dict = {'English': 'en_core_web_sm',
                    'Dutch': 'nl_core_news_sm',
                    'German': 'de_core_news_sm',
                    'French': 'fr_core_news_sm',
                    'Italian': 'it_core_news_sm',
                    'Spanish': 'es_core_news_sm', 
                    'Russian': 'ru_core_news_sm'}
    return model_dict[language]


def read_data(language) -> list:
    file_name = get_file_name(language)

    data = []
    skipped = 0
    for path in glob.glob(f'./data/Train/raw/{file_name}'):
        with open(path, 'r', encoding='utf-8') as file:
            tree = ET.parse(path)
        reviews = tree.getroot()
        for review in tqdm(reviews):
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

                if len(opinions) != 0:
                    if 'camera' in path.split("/")[-1]:
                        main_cat =  'camera'
                    else:
                        main_cat = 'restaurant'
                    labels = 'LABELS: ' + ' '.join([' '.join([f'CATEGORY1: {main_cat}', 
                                                    f'CATEGORY2: {opinion["category"]}', 
                                                    f'TARGET: {opinion["target"]}']) 
                                                    for opinion in opinions])

                    data.append((path.split('_')[-1], text, labels))
    return data


def plot_distribution(dist: Counter) -> None:
    plt.bar(dist.keys(), dist.values()) 
    plt.xticks(rotation=45)
    title = 'Percentage distribution of POS-tags'
    plt.title(title)
    plt.ylabel('Percent %')
    plt.tight_layout()
    plt.savefig('Figures/test_fig.png', format='png', )


def write_pos_dist_to_file(language: str='English', plot: str=False) -> None:
    data = read_data(language)
    model_name = get_model_name(language)
    os.system(f'python3 -m spacy download {model_name}')
    nlp = spacy.load(model_name)

    pos_tags_count = Counter([token.pos_ for datum in tqdm(data) for token in nlp(datum[1])])
    total_tags = sum(value for value in pos_tags_count.values())
    pos_tags_percantages = {key:(value/total_tags) * 100 for key, value in pos_tags_count.items()}

    with open(f'{language}_percentages.json', 'a', encoding='utf-8') as outf:
        outf.write(json.dumps(pos_tags_percantages))

    if plot:
        plot_distribution(pos_tags_percantages)

    print(pos_tags_percantages)
    print(sum(value for value in pos_tags_percantages.values()))

write_pos_dist_to_file('Dutch')
