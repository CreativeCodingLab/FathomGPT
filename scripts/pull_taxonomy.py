from fathomnet.api import taxa
from fathomnet.api import boundingboxes
import json


DEFAULT_TAXA_PROVIDER = 'fathomnet'
concepts = boundingboxes.find_concepts()[1:]

def findDescendants(concept, taxaProviderName=DEFAULT_TAXA_PROVIDER, species_only = False):
  try:
    descendants = taxa.find_taxa(taxaProviderName, concept)
  except:
    try:
        descendants = taxa.find_taxa('mbari', concept)
    except:
        return []
  return [d for d in descendants if d.rank == 'Species' or not species_only]

def findAncestors(concept, taxaProviderName=DEFAULT_TAXA_PROVIDER):
    ancestors = []
    iters = 0
    while iters < 20: # to prevent infinite loop
        try:
            parent = taxa.find_parent(taxaProviderName, concept)
        except:
            try:
                parent = taxa.find_parent('mbari', concept)
            except:
                break
        if parent.rank is None:
            break
        ancestors.append(parent)
        concept = parent.name
        iters = iters + 1
        if parent.rank.lower() == 'superkingdom':
            break
    return ancestors

def getParent(concept, taxaProviderName=DEFAULT_TAXA_PROVIDER):
    try:
        parent = taxa.find_parent(taxaProviderName, concept)
    except:
        try:
            parent = taxa.find_parent('mbari', concept)
        except:
            return ""
    return parent.name

def filterUnavailableDescendants(names):
  return [n for n in names if n.name in concepts]
  

def getRank(rank):
    if rank:
        return rank.lower()
    return ""

taxonomy = {}
with open("data/taxonomy.json") as outfile:
    taxonomy = json.load(outfile)

for c in concepts:
    if c in taxonomy:
        continue
        
    try:
        print(c, flush=True)
        descendants = filterUnavailableDescendants(findDescendants(c, species_only=False))
        ancestors = findAncestors(c)
        
        rank = ""
        for d in descendants:
            if d.rank and d.name.lower() == c.lower():
                rank = d.rank
        
        descendants = [{'name': d.name, 'rank': getRank(d.rank), 'parent': getParent(d.name)} for d in descendants if d.name.lower() != c.lower()]
        ancestors = [{'name': d.name, 'rank': getRank(d.rank)} for d in ancestors]
        ancestors.reverse()
        
        taxonomy[c] = {'rank': rank.lower(), 'taxonomy': {'descendants': descendants, 'ancestors': ancestors}}
        

        with open("data/taxonomy.json", "w") as outfile:
            json.dump(taxonomy, outfile)
    except:
        print('error', flush=True)
