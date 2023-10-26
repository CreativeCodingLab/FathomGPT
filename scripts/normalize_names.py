import json
import re
import pandas as pd


def get_singular(word):
    word = re.sub('es$', '', word)
    word = re.sub('s$', '', word)
    return word

def get_normalized(name):
    name = name.lower()
    name = name.replace('-', ' ').replace('/', ' ').replace(',', '').replace('"', '').replace('(', '').replace(')', '')
    words = name.split(' ')
    return ''.join([get_singular(w) for w in words])
    

f = open('data/names_worms.json')
names = json.load(f)

input_datapath = "data/concepts_desc.csv" 
df = pd.read_csv(input_datapath)
df = df[["concepts", "names", "links", "description", "related"]]
df = df.dropna()
for index, row in df.iterrows():
    cnames = []
    if row['names'] != ' ':
        cnames.extend(row['names'].split(', '))
    if row['links'] != ' ':
        cnames.extend(row['links'].split(', '))
    concept = row['concepts']
    for name in cnames:
        if name in names:
            if len(names[name]) > 30:
                continue
            if concept not in names[name]:
                names[name].append(concept)
        else:
            names[name] = [concept]


normalized_names = {}
for n in names:
    normalized = get_normalized(n)
    print(n+' -> '+normalized)
    if normalized not in normalized_names:
        normalized_names[normalized] = []
    normalized_names[normalized].extend(names[n])

for n in normalized_names:
    normalized_names[n] = list(set(normalized_names[n]))


with open("names_normalized.json", "w") as outfile:
    json.dump(normalized_names, outfile)
