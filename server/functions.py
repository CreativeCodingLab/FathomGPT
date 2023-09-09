from utils import *
from constants import *

from fathomnet.api import images
from fathomnet.dto import GeoImageConstraints

def findImages(concept=None, contributorsEmail=None, includeUnverified=None, includeVerified=None, limit=DEFAULT_LIMIT, maxDepth=None, minDepth=None,
    orderedBy=None, isDecending=None, isAscending=None, bodiesOfWater=None, latitude=None, longitude=None, kilometersFrom=None, milesFrom=None,
    findChildren=False, findSpeciesBelongingToTaxonomy = False, findParent=False, findClosestRelative=False, taxaProviderName=DEFAULT_TAXA_PROVIDER,
    includeGood=None, findBest=None, findWorst=None, findOtherSpecies=False, excludeOtherSpecies=False,
):
  maxLatitude, maxLongitude, minLatitude, minLongitude = latLongBoundary(latitude, longitude, kilometersFrom, milesFrom)
  
  sci_names = getScientificNames(concept)

  concept_names = []
  for name in sci_names:
    relatives = getRelatives(name, findChildren, findSpeciesBelongingToTaxonomy, findParent, findClosestRelative, taxaProviderName)
    concept_names.extend(getNamesFromTaxa(name, relatives))
  
  names = []
  if FIND_DESCENDENTS_DEFAULT:
    for name in concept_names:
      descendants = findDescendants(taxaProviderName, name, False)
      names.extend(getNamesFromTaxa(name, descendants))
      
  names = set(filterUnavailableConcepts(names))

  if DEBUG_LEVEL >= 1:
    print("Concepts to fetch: "+str(names))

  data = []
  for concept_name in names:
    data.extend(
      images.find(GeoImageConstraints(
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
      ))
    )

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
