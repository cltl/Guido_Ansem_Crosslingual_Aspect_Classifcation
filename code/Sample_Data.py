from sklearn.model_selection import train_test_split
from typing import List
import itertools
import random

from Label_Distribution import get_data

def resample_data(target_language: str,
                    aux_languages: List[str], 
                    domain: str, 
                    mode: str, 
                    n_samples:int) -> None:

    target_lang_data = get_data(domain, target_language, mode)

    aux_data = []
    for language in aux_languages:
        aux_data.append(get_data(domain, language, mode))

    size_dist = [len(lang) for lang in aux_data]
    size_dist = [size/sum(size_dist) for size in size_dist]
    aux_size = n_samples - len(target_lang_data)

    aux_data = [random.sample(aux_data[i], int(aux_size * size_dist[i])) for i in range (len(aux_data))]
    aux_data = list(itertools.chain(*aux_data))

    data = target_lang_data + aux_data

    if domain == 'Camera':
        train_data, test_data = train_test_split(data, shuffle=True, test_size=0.2, random_state=666)
        datasets = [train_data, test_data]
        names = ['train', 'test']
    else:
        datasets = [data]
        names = ['Train']

    all_languages = [target_language] + aux_languages
    for i, dataset in enumerate(datasets):
        with open(f'data/{names[i]}/{names[i]}_{domain}_{"_".join(all_languages)}_{n_samples}.txt', 
                        'a', encoding='utf-8') as outfile:

            for text, labels in dataset:
                outfile.write(' '.join([text, labels, '\n']))


target_language = 'English'
aux_languages = ['Russian']
domain = 'Restaurant'
mode = 'Train'
n_samples = 3500

resample_data(target_language, aux_languages, domain, mode, n_samples)
