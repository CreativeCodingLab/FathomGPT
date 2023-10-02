from .constants import *

from datetime import datetime
import math
from urllib.request import urlopen
from urllib.parse import quote
import json
import re

from fathomnet.api import regions
from fathomnet.api import taxa
from fathomnet.api import boundingboxes


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
    parent = taxa.find_parent(taxaProviderName, concept)
    relatives = filterUnavailableDescendants(findDescendants(parent.name, taxaProviderName, False))
    relatives = [d for d in relatives if d.name.lower() != concept.lower()]
    if len(relatives) == 0:
        concepts = boundingboxes.find_concepts()
        while parent.name not in concepts:
            try:
                parent = taxa.find_parent(taxaProviderName, parent.name)
            except:
                break
        return [parent]
    return relatives
    
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
      score = score + (box['width']*box['height'])
  if count == 0:
    return 0
  avg_size = (score/count)
  return avg_size/(d['width']*d['height'])

def filterByBoundingBoxes(data, names, includeGood, findBest, findWorst, findOtherSpecies, excludeOtherSpecies):
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
    if DEBUG_LEVEL >= 1:
        print(otherSpecies)

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

  return data


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

def getScientificNames(concept):
    f = open(NAMES_JSON)
    names = json.load(f)

    concept_normalized = get_normalized(concept)
    if DEBUG_LEVEL >= 2:
        print('normalized name: '+concept_normalized)
    if concept_normalized in names:
        return names[concept_normalized]

    return []
