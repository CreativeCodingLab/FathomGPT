from .utils import *
from .constants import *

from fathomnet.api import images
from fathomnet.dto import GeoImageConstraints

def findImages(concept=None, contributorsEmail=None, includeUnverified=None, includeVerified=None, limit=DEFAULT_LIMIT, maxDepth=None, minDepth=None,
    orderedBy=None, isDecending=None, isAscending=None, bodiesOfWater=None, latitude=None, longitude=None, kilometersFrom=None, milesFrom=None,
    findChildren=False, findSpeciesBelongingToTaxonomy = False, findParent=False, findClosestRelative=False, taxaProviderName=DEFAULT_TAXA_PROVIDER,
    includeGood=None, findBest=None, findWorst=None, findOtherSpecies=False, excludeOtherSpecies=False,
):
  maxLatitude, maxLongitude, minLatitude, minLongitude = latLongBoundary(latitude, longitude, kilometersFrom, milesFrom)
  
  names = []
  try:
    sci_names = getScientificNames(concept)
    if len(sci_names) == 0:
        sci_names = [concept]

    concept_names = []
    for name in sci_names:
        try:
            relatives = getRelatives(name, findChildren, findSpeciesBelongingToTaxonomy, findParent, findClosestRelative, taxaProviderName)
            concept_names.extend(getNamesFromTaxa(name, relatives))
        except:
            print('Failed to find relatives for '+name)
            concept_names.append(name)

    if FIND_DESCENDENTS_DEFAULT:
        for name in concept_names:
            try:
                descendants = findDescendants(taxaProviderName, name, False)
                names.extend(getNamesFromTaxa(name, descendants))
            except:
                print('Failed to find relatives for '+name)
                names.append(name)
      
    names = set(filterUnavailableConcepts(names))
  except:
    if DEBUG_LEVEL >= 1:
        print("failed to fetch concept names")

  if DEBUG_LEVEL >= 1:
    print("Concepts to fetch: "+str(names))

  data = []
  for concept_name in names:
    constraints = GeoImageConstraints(
        concept=concept_name,
        contributorsEmail=contributorsEmail,
        includeUnverified=includeUnverified,
        includeVerified=includeVerified,
        limit=RETRIEVAL_LIMIT,
        maxDepth=maxDepth,
        maxLatitude=maxLatitude,
        maxLongitude=maxLongitude,
        minDepth=minDepth,
        minLatitude=minLatitude,
        minLongitude=minLongitude,
        taxaProviderName=taxaProviderName
      )
    data.extend(
      images.find(constraints)
    )

  if DEBUG_LEVEL >= 1:
    print("Num images retrieved: "+str(len(data)))
  if len(data) == 0:
    return []
  data = [r.to_dict() for r in data]
  data = removeDuplicateImages(data)

  data = filterByBodiesOfWater(data, bodiesOfWater)
  
  data = filterByBoundingBoxes(data, names, includeGood, findBest, findWorst, findOtherSpecies, excludeOtherSpecies)

  if orderedBy is not None:
    data = sort_images(data, orderedBy.lower(), isAscending, isDecending)

  return data[:limit]

def count_all():
  return images.count_all()

def count_by_submitter(contributors_email):
  return images.count_by_submitter(contributors_email)
