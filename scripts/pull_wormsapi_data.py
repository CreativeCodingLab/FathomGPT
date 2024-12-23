import json
import time
from urllib.request import urlopen
from urllib.parse import quote


WORMS_URL = 'https://fathomnet.org/worms/taxa/info/'

f = open('data/concepts.json')
concepts = json.load(f)

names = {}
for concept in concepts:
    print(concept)
    s = concept.lower()
    if s not in names:
        names[s] = []
    names[s].append(concept)
    
    try:
        response = urlopen(WORMS_URL + quote(concept))
        synonyms = json.loads(response.read())['alternateNames']
        for s in synonyms:
            s = s.lower()
            if s not in names:
                names[s] = []
            names[s].append(concept)
    except Exception as e:
        print("  "+str(e))
        continue
    
    time.sleep(0.01)

for n in names:
    names[n] = list(set(names[n]))

with open("names_worms.json", "w") as outfile:
    json.dump(names, outfile)
