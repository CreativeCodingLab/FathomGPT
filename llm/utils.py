from .constants import *

from datetime import datetime
import math
from urllib.request import urlopen
from urllib.parse import quote
import json
import re
import random
from difflib import SequenceMatcher

from fathomnet.api import regions
from fathomnet.api import taxa
from fathomnet.api import boundingboxes
from fathomnet.api import images
from fathomnet.dto import GeoImageConstraints

import openai

openai.api_key = KEYS['openai']


# Sorting ---------------------------

def toTimestamp(d):
  if '.' in d['timestamp']:
    return datetime.strptime(d['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
  else:
    return datetime.strptime(d['timestamp'], "%Y-%m-%dT%H:%M:%SZ")

def sort_images(data, orderedBy, isAscending, isDecending):
    for kw in ['oldest', 'newest', 'added', 'created']:
        if kw in orderedBy or orderedBy == 'date':
            data.sort(key=lambda d: toTimestamp(d))
            if isAscending or 'oldest' in orderedBy:
                return data
            data.reverse()
            return data

    if 'updated' in orderedBy:
        data.sort(key=lambda d: datetime.strptime(d['lastUpdatedTimestamp'], "%Y-%m-%dT%H:%M:%S.%fZ"))

    if 'size' in orderedBy:
        data.sort(key=lambda d: d['width']*d['height'])

    if orderedBy in data[0]:
        data.sort(key=lambda d: d[orderedBy])

    keys = list(data[0].keys())
    keys = [k for k in keys if orderedBy in k.lower() and data[0][k] is not None]
    if len(keys) > 0:
        data.sort(key=lambda d: d[keys[0]])

    if isAscending:
        return data
    data.reverse()
    return data

def removeDuplicateImages(data):
    seen = []
    data_filtered = []
    for d in data:
        if d['uuid'] not in seen:
            data_filtered.append(d)
            seen.append(d['uuid'])
    return data_filtered

# Location filtering ---------------------------

# algorithm from https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
def latLongOffsetAppox(lat, lon, distance, direction):
    # offsets in meters
    dn = 0
    de = 0
    if direction == 'n':
        dn = distance
    elif direction == 's':
        dn = -distance
    elif direction == 'e':
        de = distance
    elif direction == 'w':
        de = -distance
    else:
        print("Invalid direction")  # Handle invalid direction

    # Coordinate offsets in radians
    dLat = dn/EARTH_RADIUS
    dLon = de/(EARTH_RADIUS*math.cos(math.pi*lat/180))

    # OffsetPosition, decimal degrees
    latO = lat + dLat * 180/math.pi
    lonO = lon + dLon * 180/math.pi

    return latO, lonO

def latLongBoundary(lat, lon, kilometersFrom, milesFrom):
    if kilometersFrom is not None:
        distMeters = kilometersFrom * 1000
    elif milesFrom is not None:
        distMeters = milesFrom * MILES_TO_METERS
    else:
        return None, None, None, None

    latN, _ =  latLongOffsetAppox(lat, lon, distMeters, 'n')
    latS, _ =  latLongOffsetAppox(lat, lon, distMeters, 's')
    _, longE =  latLongOffsetAppox(lat, lon, distMeters, 'e')
    _, longW =  latLongOffsetAppox(lat, lon, distMeters, 'w')

    return max(latN, latS), max(longE, longW), min(latN, latS), min(longE, longW)

def filterByBodiesOfWater(data, bodiesOfWater):
  if bodiesOfWater is None:
    return data

  available_regions = regions.find_all()
  latLong = []
  for region in bodiesOfWater.split(", "):
      latLong.append(next(r for r in available_regions if region == r.name))

  filtered_data = []
  for d in data:
    if d['longitude'] is None:
      continue
    for r in latLong:
      # todo: double check if this wrap-around calculation is correct
      # eg. pacific ocean has longitude 60 to -120. Monterey bay has longitude -130
      # this calculation changes -120 to 240 and -130 to 230
      maxLon = r.maxLongitude
      dataLon = d['longitude']
      if maxLon < r.minLongitude:
        maxLon = maxLon % 180 + 180
        if dataLon < 0:
          dataLon = dataLon % 180 + 180

      if r.minLatitude <= d['latitude'] <= r.maxLatitude and r.minLongitude <= dataLon <= maxLon:
        filtered_data.append(d)

  return filtered_data

# Taxonomy ---------------------------

def findDescendants(concept, taxaProviderName=DEFAULT_TAXA_PROVIDER, species_only = True):
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

def findRelatives(concept, taxaProviderName=DEFAULT_TAXA_PROVIDER):
    ancestors = findAncestors(concept, taxaProviderName)
    descendants = taxa.find_taxa(taxaProviderName, concept)
    descendants = [d for d in descendants if concept.lower() and d.name.lower()]
    rank = 'Species'
    if len(descendants) > 0:
        rank = descendants[0].rank
    
    for a in ancestors:
        relatives = filterUnavailableDescendants(findDescendants(a.name, taxaProviderName, False))
        relatives = [d for d in relatives if d.name.lower() != concept.lower() and d.name.lower() != a.name.lower() and d.rank == rank]
        if len(relatives) > 0:
            return relatives
    if len(ancestors) == 0:
        return []
    return [ancestors[0].name]

    
def getParent(concept, taxaProviderName=DEFAULT_TAXA_PROVIDER):
    try:
        parent = taxa.find_parent(taxaProviderName, concept)
    except:
        try:
            parent = taxa.find_parent('mbari', concept)
        except:
            return ""
    return parent.name

def getRelatives(concept, findChildren, findSpeciesBelongingToTaxonomy, findParent, findClosestRelative, taxaProviderName):
  if findChildren:
    return taxa.find_children(taxaProviderName, concept)
  if findSpeciesBelongingToTaxonomy:
    return findDescendants(parent, taxaProviderName)
  if findParent:
    return [taxa.find_parent(taxaProviderName, concept)]
  if findClosestRelative:
    return findRelatives(concept, taxaProviderName)
  return None

def getNamesFromTaxa(concept, taxa):
  if taxa is None:
    return [concept]
  taxa = [t.name for t in taxa if t.name != concept]
  if len(taxa) == 0:
    return [concept]
  return taxa
  
def filterUnavailableConcepts(names):
  concepts = boundingboxes.find_concepts()
  return [n for n in names if n in concepts]

def filterUnavailableDescendants(names):
  concepts = boundingboxes.find_concepts()
  return [n for n in names if n.name in concepts]

# Bounding box processing ---------------------------

def containsOtherSpecies(boundingBoxes, names):
  for box in boundingBoxes:
    if box['concept'] not in names:
      return True
  return False

def marginGood(margin_width, image_width):
    return margin_width / image_width > GOOD_BOUNDING_BOX_MIN_MARGINS

def boundingBoxQualityScore(d, names):
  # the score is the average size of the bounding boxes divided by the image size
  score = 0
  count = 0
  if d['width']*d['height'] == 0:
    return 0
  for box in d['boundingBoxes']:
    if box['concept'] not in names:
      continue
    count = count + 1
    if marginGood(box['x'], d['width']) and marginGood(d['width']-(box['x']+box['width']), d['width']) \
      and marginGood(box['y'], d['height']) and marginGood(d['height']-(box['y']+box['height']), d['width']):
      # boxes that fill the entire image have score=0
      s = (box['width']*box['height'])
      if s > score:
        score = s
  if count == 0:
    return 0
  #avg_size = (score/count)
  #return avg_size/(d['width']*d['height'])
  return score/(d['width']*d['height'])

def filterByBoundingBoxes(data, names, includeGood, findBest, findWorst, findOtherSpecies, excludeOtherSpecies):
  metadata = {}
  if findOtherSpecies:
    data = [d for d in data if containsOtherSpecies(d['boundingBoxes'], names)]
    otherSpecies = {}
    for d in data:
      for b in d['boundingBoxes']:
        other = b['concept']
        if other in names:
          continue
        if other not in otherSpecies:
          otherSpecies[other] = 0
        otherSpecies[other] = otherSpecies[other] + 1
    print(otherSpecies)
    metadata = {'others': otherSpecies}
    scores = {}
    for d in data:
      scores[d['uuid']] = boundingBoxQualityScore(d, otherSpecies.keys())
    data.sort(key=lambda d: scores[d['uuid']], reverse=True)
    

  elif excludeOtherSpecies:
    data = [d for d in data if not containsOtherSpecies(d['boundingBoxes'], names)]
  
  if includeGood or findBest or findWorst:
    scores = {}
    for d in data:
      scores[d['uuid']] = boundingBoxQualityScore(d, names)
    data = [d for d in data if scores[d['uuid']] > 0]
    
    if includeGood:
        data = [d for d in data if scores[d['uuid']] > GOOD_BOUNDING_BOX_MIN_SIZE]
    if findBest:
        data.sort(key=lambda d: scores[d['uuid']], reverse=True)
    if findWorst:
      data.sort(key=lambda d: scores[d['uuid']])

  return data, metadata


# Name ---------------------------

def get_singular(word):
    word = re.sub('es$', '', word)
    word = re.sub('s$', '', word)
    return word

def get_normalized(name):
    name = name.lower()
    name = name.replace('-', ' ').replace('/', ' ').replace(',', '').replace('"', '').replace('(', '').replace(')', '')
    words = name.split(' ')
    return ''.join([get_singular(w) for w in words])
    
def isNameAvaliable(concept):
    concepts = boundingboxes.find_concepts()
    return concept in concepts

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()
    
def getScientificNames(concept, should_normalize=True, filename=NAMES_JSON, is_dict=False):
    f = open(filename)
    names = json.load(f)

    if should_normalize:
        concept = get_normalized(concept)

    if DEBUG_LEVEL >= 2:
        print('normalized name: '+concept)
    if concept in names:
        if is_dict:
            return list(names[concept].keys())
        return names[concept]
        
    concepts = []
    if len(concept) > 3:
        for name in names:
            if similarity(name, concept) > SCI_NAME_MATCH_SIMILARITY:
                if is_dict:
                    concepts.extend(list(names[name].keys()))
                else:
                    concepts.extend(names[name])
                    
            if not should_normalize and concept in name.split(' '):
                concepts.extend(names[name])

    return concepts


# ==== post-processing langchain ====

def findImages(includeOnlyGood=False, findBest=False, findWorst=False, findOtherSpecies=False, excludeOtherSpecies=False):
    #print('called')
    return {'includeGood': includeGood, 'findBest': findBest, 'findWorst': findWorst, 'findOtherSpecies': findOtherSpecies, 'excludeOtherSpecies': excludeOtherSpecies}


def run_chatgpt(prompt):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        functions=[
            {
                "name": "findImages",
                "description": "Get a constrained list of images.",
                "parameters": {
                    "type": "object",
                    "properties": FUNCTION_PROPERTIES,
                    "required": []
                }
            }
        ],
        function_call="auto",
    )

def changeNumberToFetch(sql):
    try:
        limit = int(re.search('SELECT TOP ([0-9]+) ', sql, re.IGNORECASE).group(1))
        sql = re.sub('SELECT TOP [0-9]+ ', '', sql, re.IGNORECASE)
        sql = 'SELECT TOP '+str(RETRIEVAL_LIMIT)+' '+sql
        #print(limit)
        #print(sql)
        return limit, sql
    except:
        return -1, sql


def getProp(props, key):
    if key in props:
        return props[key]
    return False

def findByURL(url, results):
    for r in results:
        if r['url'] == url:
            return r
    return None

def noNulls(r):
    for key in r:
        if r[key] is None:
            return False
    return True

def postprocess(results, limit, prompt, sql):
    #print(results[0])
    if not isinstance(results, list) or len(results) == 0 or 'url' not in results[0]:
        return results
    
    deduped = []
    urls = []
    for r in results:
        if r['url'] not in urls:
            deduped.append(r)
            urls.append(r['url'])
    results = deduped

    results = [r for r in results if noNulls(r)]

    concepts = boundingboxes.find_concepts()[1:]
    concepts = [c for c in concepts if sql.count(c) > 0]
    if 'concept' in results[0]:
        concepts = set([d['concept'] for d in results if d['concept'] in concepts])
    urls = {d['url']:'' for d in results}
    
    #print(concepts)
    
    data = []
    for concept in concepts:
        constraints = GeoImageConstraints(concept=concept)
        imgs = images.find(constraints)
        imgs = [d.to_dict() for d in imgs]
        data.extend([findByURL(url, imgs) for url in urls])
    data = [d for d in data if d is not None]
    if len(data) == 0:
        return results
    

    #print(prompt)
    chatResponse = run_chatgpt(prompt)
    props = json.loads(chatResponse.choices[0].message.function_call.arguments)
    #print(props)
    
    data, metadata = filterByBoundingBoxes(data, concepts, getProp(props, 'includeGood'), getProp(props, 'findBest'), getProp(props, 'findWorst'), getProp(props, 'findOtherSpecies'), getProp(props, 'excludeOtherSpecies'))
    
    
    #print(json.dumps(data[:5]))

    urls = {d['url']: '' for d in data}
    results = [findByURL(url, results) for url in urls]
    results = [r for r in results if r is not None]
    
    if getProp(props, 'findOtherSpecies') and 'others' in metadata:
        for r in results:
            d = findByURL(r['url'], data)
            for b in d['boundingBoxes']:
                if b['concept'] in metadata['others']:
                    r['concept'] = b['concept']
                    break
    
    if getProp(props, 'orderedBy') == 'random':
        random.shuffle(results)

    if limit == -1 and len(results) > 10:
        return results[:10]
        
    return results
