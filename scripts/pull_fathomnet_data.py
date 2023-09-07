from fathomnet.api import images
from fathomnet.api import boundingboxes
from fathomnet.dto import GeoImageConstraints
import json
import os
import time

concepts = boundingboxes.find_concepts()[1:]

with open("concepts.json", "w") as outfile:
    json.dump(concepts, outfile)

if not os.path.exists("data"):
    os.makedirs("data")

offset = 1105 
for concept in concepts[offset:]:
    print(concept)
    data = images.find(GeoImageConstraints(
        concept=concept,
        contributorsEmail=None,
        includeUnverified=None,
        includeVerified=None,
        limit=None,
        maxDepth=None,
        maxLatitude=None,
        maxLongitude=None,
        minDepth=None,
        minLatitude=None,
        minLongitude=None,
        taxaProviderName=None
    ))
    data = [r.to_dict() for r in data]
    
    concept = concept.replace('/', ' ').replace('"', '')
    with open("data/"+concept+".json", "w") as outfile:
        json.dump(data, outfile)
    time.sleep(0.1)
