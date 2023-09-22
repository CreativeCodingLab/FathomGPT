import pandas as pd
import numpy as np
import wikipedia
from fathomnet.api import taxa

MIN_DESC_LEN = 1
MIN_DESC_SENTENCES = 0
ITERS_TEST = 10000
taxaProviderName = 'fathomnet'
input_datapath = "data/concepts_desc.csv"
df = pd.read_csv(input_datapath)
df = df[["concepts", "description", "related"]]
print(df.head(2))

def isBad(desc):
    return len(desc) < MIN_DESC_LEN or desc.count('.') < MIN_DESC_SENTENCES

for i, row in df.iterrows():
    if i>ITERS_TEST:
        break
    desc = row['description']
    orig_desc = desc
    concept = row['concepts']
    while True:
        if str(desc) == 'nan':
            desc = ""
        if isBad(desc):
            print(concept)
            try:
                parent = taxa.find_parent(taxaProviderName, concept).name
                if concept == 'object' or parent == 'object' or parent == 'equipment':
                    break
                concept = parent
            except:
                break
            related = wikipedia.search(concept, results=5)
            name = concept
            if len(related) > 0:
                related = [r for r in related if r.count('List ')==0] + [r for r in related if r.count('List ')>0]
                name = related[0]
            try:
                cat = wikipedia.page(name).categories
                if ('All stub articles' in cat and 'Short description matches Wikidata' in cat) or "Articles with 'species' microformats" not in cat:
                    continue
                summary = wikipedia.summary(name, auto_suggest=False, sentences=2)
                summary = summary.split('\n')[0]
                if isBad(summary):
                    continue
                print(concept)
                if str(orig_desc) == 'nan':
                    desc = summary
                else:
                    desc = orig_desc+" "+summary
                df.at[i,'description'] = desc
                if str(row['related']) == 'nan' and len(related) > 0:
                    df.at[i,'related'] = ", ".join(related)
                
            except:
                continue
        else:
            break

for i, row in df.iterrows():
    if str(row['description']) == 'nan': 
        df.at[i,'description'] = row['concepts']
    if str(row['related']) == 'nan': 
        df.at[i,'related'] = ' '

print(df.at[9, 'description'])
print(df.head(2))
df.to_csv("data/concepts_desc2.csv")
