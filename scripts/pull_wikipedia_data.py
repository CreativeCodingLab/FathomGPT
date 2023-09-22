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
        related = wikipedia.search(concept, results=5)
        if len(related) > 0:
            related = [r for r in related if r.count('List ')==0] + [r for r in related if r.count('List ')>0] # move "List ..." to the back
            name = related[0]
        cat = wikipedia.page(name).categories
        """
        if "Articles with 'species' microformats" not in cat:
            related2 = wikipedia.search(concept+" genus", results=5)
            if len(related2) > 0:
                related = [r for r in related2 if r.count('List ')==0] + [r for r in related2 if r.count('List ')>0]
                name = related[0]
                cat = wikipedia.page(name).categories
        """
        
        if "Articles with 'species' microformats" not in cat:
            description.append("")
            related_terms.append("")
            continue
            
        if name != concept:
            related_terms.append(", ".join(related))
        elif len(related)>1:
            related_terms.append(", ".join(related[1:]))
        else:
            related_terms.append("")
    except:
        related_terms.append("")
        
    try:
        summary = wikipedia.summary(name, auto_suggest=False, sentences=2)
        summary = summary.split('\n')
        description.append(summary[0])
    except:
        description.append("")


df['description'] = description
df['related'] = related_terms
df.to_csv("data/concepts_desc.csv")


#print(wikipedia.search("Octogonade mediterranea"))
#print(wikipedia.summary("Bathypolypus", sentences=2).split("\n"))
#print(wikipedia.page("Aurelia aurita").categories)
