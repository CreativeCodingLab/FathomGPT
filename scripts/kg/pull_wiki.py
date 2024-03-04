import pandas as pd
import wikipedia
import json
from fathomnet.api import taxa


def isBiological(page):
    cat = page.categories
    if "Articles with 'species' microformats" in cat:
        return True
    summary = page.summary.lower()
    if summary.count('species')>0 or summary.count('genus')>0 or summary.count('family')>0 or summary.count('order')>0 or summary.count('class')>0 or summary.count('phylum')>0:
        return True
    return False
    
def sanitizeConcept(concept):
    if concept.find(' sp.')!=-1:
        concept = concept[:concept.find(' sp.')]
    if concept.find(' (')!=-1 and concept.find(')')!=-1:
        concept = concept[:concept.find(' (')]+concept[concept.find(')')+1:]
    concept = concept.replace(' cf. ', ' ')
    return concept


input_datapath = "data/concepts.csv"
df = pd.read_csv(input_datapath)
df = df.dropna()

with open('scripts/kg/descriptions.json') as f:
    data = json.load(f)
i = 0
for concept in df.concepts:
    print(concept)
    concept = sanitizeConcept(concept)
    
    if concept in data:
        continue
    
    try:
        page = wikipedia.page(concept, auto_suggest=False)
    except:
        continue
        
    if not isBiological(page):
        continue
    
    data[concept] = {'wiki': page.content}
    
    if i%100 == 0:
        with open("scripts/kg/descriptions.json", "w") as outfile:
            json.dump(data, outfile)
    i = i+1
    
    #print(page.content)
    #break
    
with open("scripts/kg/descriptions.json", "w") as outfile:
    json.dump(data, outfile)
