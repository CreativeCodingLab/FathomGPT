from utils import *
from constants import *

from fathomnet.api import images
from fathomnet.dto import GeoImageConstraints

def findImages(concept=None, contributorsEmail=None, includeUnverified=None, includeVerified=None, limit=DEFAULT_LIMIT, maxDepth=None, minDepth=None,
    orderedBy=None, isDecending=None, isAscending=None, bodiesOfWater=None, latitude=None, longitude=None, kilometersFrom=None, milesFrom=None,
    findChildren=False, findSpeciesBelongingToTaxonomy = False, findParent=False, findClosestRelative=False, taxaProviderName=DEFAULT_TAXA_PROVIDER,
    includeGood=None, findOtherSpecies=False, excludeOtherSpecies=False,
):
  concept = getScientificName(concept)

  maxLatitude, maxLongitude, minLatitude, minLongitude = latLongBoundary(latitude, longitude, kilometersFrom, milesFrom)

  relatives = getRelatives(concept, findChildren, findSpeciesBelongingToTaxonomy, findParent, findClosestRelative, taxaProviderName)
  if relatives is not None:
    species_names = [t.name for t in relatives if t.name != concept]
  else:
    species_names = [concept]

  data = []
  for species in species_names:
    data.extend(
      images.find(GeoImageConstraints(
        concept=species,
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

  data = filterByBodiesOfWater(data, bodiesOfWater)
  
  data = filterByBoundingBoxes(data, concept, includeGood, findOtherSpecies, excludeOtherSpecies)

  if orderedBy is not None:
    data = sort_images(data, orderedBy.lower(), isAscending, isDecending)

  return data[:limit]

def count_all():
  return images.count_all()

def count_by_submitter(contributors_email):
  return images.count_by_submitter(contributors_email)
