import glob
from collections import Counter
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt

camera_labels = ['IMAGING', 'GENERAL', 'OTHER', 'PRICE', 'DISPLAY', 'EXPOSURE', 'DIMENSION', 'LENS', 'BATTERY', 'VIDEO', 'ZOOM', 'PERFORMANCE', 'MEMORY']
restaurant_labels = ['RESTAURANT#GENERAL', 'NULL', 'SERVICE#GENERAL', 'AMBIENCE#GENERAL', 'FOOD#QUALITY', 'FOOD#PRICES', 'RESTAURANT#PRICES', 'FOOD#STYLE_OPTIONS', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS', 'LOCATION#GENERAL', 'DRINKS#PRICES', 'RESTAURANT#MISCELLANEOUS']

data = []
skipped = 0
for path in glob.glob('./data/Test/raw/Res*.B'):
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

languages = [datum[0].split('.')[0] for datum in data]
languages = Counter(languages)

fig, ax = plt.subplots()
plt.bar(languages.keys(), languages.values()) 
plt.xticks(rotation=45)
plt.ylabel('Number of Samples', fontsize=12)
for i, v in enumerate(languages.values()):
    ax.text(x=i-.35, y=v+1, s=str(v), color='black', fontsize=12)
title = 'Restaurant Test All Languages'
plt.title(title, fontsize=16)
plt.tight_layout()
plt.savefig(f'Figures/Restaurant_Test_Language_Dist.png', format='png', )

x = 0