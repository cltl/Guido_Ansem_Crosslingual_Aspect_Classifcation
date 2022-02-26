import glob
import json
import numpy as np
import matplotlib.pyplot as plt


dists = {}
for path in glob.glob(f'./*percentages*'):
    with open(path, 'r', encoding='utf-8') as f:
        dists[path.split('_')[0].split('/')[-1]] = json.load(f)


all_pos_tags = set()
for language in dists.keys():
    for tag in dists[language]:
        all_pos_tags.add(tag)

for language in dists.keys():
    for tag in all_pos_tags:
        if tag not in dists[language]:
            dists[language][tag] = 0

# set width of bars
barWidth = 1/(len(dists) + 4)
 
# set heights of bars
dutch_height = [value for value in dists['Dutch'].values()]
english_height = [value for value in dists['English'].values()]
german_height = [value for value in dists['German'].values()]
italian_height = [value for value in dists['Italian'].values()]
spanish_height = [value for value in dists['Spanish'].values()]
french_height = [value for value in dists['French'].values()]
russian_height = [value for value in dists['Russian'].values()]
# turkish_height = [value for value in dists['Turkish'].values()]
 
# Set position of bar on X axis
r1 = np.arange(len(dutch_height))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]
# r8 = [x + barWidth for x in r7]
 
# Make the plot
f, ax = plt.subplots(figsize=(18,5))
plt.bar(r1, dutch_height, color='#B7302C', width=barWidth, edgecolor='white', label='Dutch')
plt.bar(r2, english_height, color='#B5B72C', width=barWidth, edgecolor='white', label='English')
plt.bar(r3, german_height, color='#3DB72C', width=barWidth, edgecolor='white', label='German')
plt.bar(r4, italian_height, color='#2CB7AA', width=barWidth, edgecolor='white', label='Italian')
plt.bar(r5, spanish_height, color='#B52CB7', width=barWidth, edgecolor='white', label='Spanish')
plt.bar(r6, french_height, color='#CD2A04', width=barWidth, edgecolor='white', label='French')
plt.bar(r7, russian_height, color='#2C3BB7', width=barWidth, edgecolor='white', label='Russian')
# plt.bar(r8, turkish_height, color='#000000', width=barWidth, edgecolor='white', label='Turkish')
 
# Add xticks on the middle of the group bars
plt.xlabel('POS tags', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(dutch_height))], dists['Dutch'].keys(), rotation=45)
 
# Create legend & Show graphic
plt.legend()
plt.tight_layout()
plt.savefig(f'Figures/pos_tags.png', format='png', )
x = 0