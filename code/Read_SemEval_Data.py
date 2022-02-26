import glob
import random
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


data = []
skipped = 0
for path in glob.glob('./data/Train/raw/cam*en.xml'):
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

                data.append((text, labels))


train_data, test_data = train_test_split(data, shuffle=True, test_size=0.2, random_state=666)

datasets = [train_data, test_data]
names = ['train', 'test']

for i, dataset in enumerate(datasets):
    with open(f'data/{names[i]}/{names[i]}_camera_english.txt', 'a', encoding='utf-8') as outfile:
        for text, labels in dataset:
            outfile.write(' '.join([text, labels, '\n']))

print(skipped)
print('Done!')
