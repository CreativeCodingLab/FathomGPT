from constants import *

from datetime import datetime
import math
from urllib.request import urlopen
from urllib.parse import quote
import json

from fathomnet.api import regions
from fathomnet.api import taxa


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

def findDescendantSpecies(taxaProviderName, concept):
  descendants = taxa.find_taxa(taxaProviderName, concept)
  return [d for d in descendants if d.rank == 'Species']

def getRelatives(concept, findChildren, findSpeciesBelongingToTaxonomy, findParent, findClosestRelative, taxaProviderName):
  if findChildren:
    return taxa.find_children(taxaProviderName, concept)
  if findSpeciesBelongingToTaxonomy:
    return findDescendantSpecies(taxaProviderName, parent)
  if findParent:
    return [taxa.find_parent(taxaProviderName, concept)]
  if findClosestRelative:
    parent = taxa.find_parent(taxaProviderName, concept).name
    return findDescendantSpecies(taxaProviderName, parent)
  return None


# Bounding box processing ---------------------------

def containsOtherSpecies(boundingBoxes, concept):
  for box in boundingBoxes:
    if box['concept'] != concept:
      return True
  return False

def boundingBoxIsGood(d, concept):
  # return true if there is at least 1 large enough bounding box of the concept
  for box in d['boundingBoxes']:
    if box['concept'] != concept:
      continue
    if box['width']/d['width'] > GOOD_BOUNDING_BOX_SIZE and box['height']/d['height'] > GOOD_BOUNDING_BOX_SIZE:
      return True
  return False

def filterByBoundingBoxes(data, concept, includeGood, findOtherSpecies, excludeOtherSpecies):
  if findOtherSpecies:
    data = [d for d in data if containsOtherSpecies(d['boundingBoxes'], concept)]
    otherSpecies = {}
    for d in data:
      for b in d['boundingBoxes']:
        other = b['concept']
        if other == concept:
          continue
        if other in otherSpecies:
          otherSpecies[other] = otherSpecies[other] + 1
        else:
          otherSpecies[other] = 1
    if DEBUG_LEVEL >= 1:
        print(otherSpecies)

  elif excludeOtherSpecies:
    data = [d for d in data if not containsOtherSpecies(d['boundingBoxes'], concept)]
  
  if includeGood:
    data = [d for d in data if boundingBoxIsGood(d, concept)]

  return data


# Name ---------------------------

def getScientificName(concept):
  concept = concept.lower()
  if DEBUG_LEVEL >= 2:
    print(WORMS_URL + quote(concept))
  try:
    response = urlopen(WORMS_URL + quote(concept))
    synonyms = json.loads(response.read())
  except Exception as e:
    if DEBUG_LEVEL >= 2:
        print(e)
    return concept.capitalize()
  else:
    if len(synonyms) == 0 or ('code' in synonyms and synonyms['code'] == 404):
      return concept.capitalize()
    return synonyms[0]
