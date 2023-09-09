import json
import re


def get_singular(word):
    word = re.sub('es$', '', word)
    word = re.sub('s$', '', word)
    return word

def get_normalized(name):
    name = name.lower()
    name = name.replace('-', ' ').replace('/', ' ').replace(',', '').replace('"', '').replace('(', '').replace(')', '')
    words = name.split(' ')
    words.sort()
    return ' '.join([get_singular(w) for w in words])
    

f = open('data/names.json')
names = json.load(f)

normalized_names = {}
for n in names:
    normalized = get_normalized(n)
    print(n+' -> '+normalized)
    if normalized not in normalized_names:
        normalized_names[normalized] = []
    normalized_names[normalized].extend(names[n])

with open("names_normalized.json", "w") as outfile:
    json.dump(normalized_names, outfile)
