# Save data from wikipedia for each concept and its genus. 
# If the page doesn't exist or is very short, use its ancestor's wiki page (excluding 1st paragraph), replacing the ancestor name with the concept name
# Sanitize the concept names (eg. "Acanthascinae sp. 1-4 complex", "Cirrata \"egg\"", "Holothuroidea/Actiniaria", "Polychelidae 2", "Spongosorites (right)", "Uroptychus n.", "Aspidodiadema cf. hawaiiense")
# What to do with cases like "1-gallon paint bucket" if there's no wiki page? Ignore them? Use the last word (eg. "bucket", "laser")?

import pandas as pd
import wikipedia


input_datapath = "data/concepts.csv"
df = pd.read_csv(input_datapath)
df = df.dropna()
print(df.head(2))

description = []
related_terms = []
for concept in df.concepts:
    print(concept)
    name = concept
    try:
        related = wikipedia.search(concept)
        if len(related) > 0:
            name = related[0]
        if name != concept:
            related_terms.append(", ".join(related))
        elif len(related)>1:
            related_terms.append(", ".join(related[1:]))
        else:
            related_terms.append("")
    except:
        related_terms.append("")
        
    try:
        summary = wikipedia.summary(name, sentences=2)
        summary = summary.split('\n')
        description.append(summary[0])
    except:
        description.append("")


df['description'] = description
df['related'] = related_terms
df.to_csv("data/concepts_desc.csv")


#print(wikipedia.search("Octogonade mediterranea"))
#print(wikipedia.summary("Bathypolypus", sentences=2).split("\n"))

