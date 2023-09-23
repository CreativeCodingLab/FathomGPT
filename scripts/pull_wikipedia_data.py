import pandas as pd
import wikipedia
from fathomnet.api import taxa


MIN_DESC_LEN = 1
MAX_DESC_LEN = 200
MIN_DESC_SENTENCES = 0
taxaProviderName = 'fathomnet'
ITERS_TEST = 10000

def isGood(desc, cat, isVeryGood = False):
    if len(desc) < MIN_DESC_LEN or desc.count('.') < MIN_DESC_SENTENCES:
        return False
    if isVeryGood:
        return 'All stub articles' not in cat or 'Good articles' in cat
    return 'All stub articles' not in cat or 'Good articles' in cat or 'Short description is different from Wikidata' in cat

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
    

def getConceptData(concept):
    name = concept
    related = wikipedia.search(concept, results=5)
    if len(related) > 0:
        related = [r for r in related if r.count('List ')==0] + [r for r in related if r.count('List ')>0] # move "List ..." to the back
        if related[0].count("List ") == 0:
            name = related[0]
        
    try:
        page = wikipedia.page(name, auto_suggest=False)
    except:
        return "", "", None
        
    if not isBiological(page):
        return "", "", page
        
    if len(related)>1:
        related = ", ".join(related[1:])
    else:
        related = ""

    summary = page.summary
    summary = summary.split('\n')[0]
    sentences = summary.split('.')
    summary = sentences[0].strip()+"."
    if len(sentences) > 1 and sentences[1].strip() != '':
        summary = summary + " " + sentences[1].strip()+"."
    
    return summary, related, page

    
def getAncestorData(concept, desc, related_terms):
    while True:
        if len(desc) > MAX_DESC_LEN:
            break
        
        try:
            parent = taxa.find_parent(taxaProviderName, concept).name
        except:
            print(' none')
            break
        print(' '+parent)
        if parent == concept or concept == 'object' or parent == 'object' or parent == 'equipment':
            if desc == '':
                desc = concept+' '+parent
            break
        concept = parent
        
        
        summary, related, page = getConceptData(concept)
        if page is None:
            continue
        
        if isGood(summary, page.categories) or desc == '':
            print("- "+concept)
            if desc == '':
                desc = summary
            else:
                desc = desc+" "+summary

            if related_terms == '' and len(related) > 0:
                related_terms = related
            if isGood(summary, page.categories, True):
                break
    return desc, related_terms
    

input_datapath = "data/concepts.csv"
df = pd.read_csv(input_datapath)
df = df.dropna()
print(df.head(2))

description = []
related_terms = []
i = 0
for concept in df.concepts:
    if i>ITERS_TEST:
        description.append("")
        related_terms.append("")
        continue

    print(concept)
    concept = sanitizeConcept(concept)
    
    summary, related, page = getConceptData(concept)

    if page is None or not isGood(summary, page.categories, True):
        summary, related = getAncestorData(concept, summary, related)
        
    if summary == '': 
        summary = concept
    if related == '': 
        related = ' '
    
    description.append(summary)
    related_terms.append(related)
    i = i+1


df['description'] = description
df['related'] = related_terms
df.to_csv("data/concepts_desc3.csv")
