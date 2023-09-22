import pandas as pd
import wikipedia
from fathomnet.api import taxa

taxaProviderName = 'fathomnet'
input_datapath = "data/concepts_desc.csv"
df = pd.read_csv(input_datapath)
df = df.dropna()
print(df.head(2))

for row in df.iterrows():
    if row['description'] == '':
        concept = row['concept']
        while True:
            try:
                parent = taxa.find_parent(taxaProviderName, concept)
            except:
                break
            related = wikipedia.search(concept)
            name = concept
            if len(related) > 0:
                name = related[0]
            try:
                summary = wikipedia.summary(name, sentences=2)
                summary = summary.split('\n')[0]
                row['description'] = summary
                if row['related'] == "" and len(related) > 0:
                    row['related'] = ", ".join(related)
                break
            except:
                concept = parent

df.to_csv("data/concepts_desc2.csv")
