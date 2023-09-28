import openai
import json
from datetime import datetime
import math
import os


# load and set our key
openai.api_key = os.getenv("OPENAI_API_KEY")

from fathomnet.api import images
from fathomnet.api import regions
from fathomnet.dto import GeoImageConstraints

DEFAULT_LIMIT = 5
RETRIEVAL_LIMIT = 100

EARTH_RADIUS=6378137 # Earth’s radius, sphere
MILES_TO_METERS=1609.34

def findImages(concept=None, contributorsEmail=None, includeUnverified=None, includeVerified=None, limit=DEFAULT_LIMIT, maxDepth=None, minDepth=None, taxaProviderName=None,
    orderedBy=None, includeGood=None, isDecending=None, isAscending=None, bodiesOfWater=None, latitude=None, longitude=None, kilometersFrom=None, milesFrom=None
):
  maxLatitude, maxLongitude, minLatitude, minLongitude = latLongBoundary(latitude, longitude, kilometersFrom, milesFrom)

  data = images.find(GeoImageConstraints(
      concept=concept,
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

  if len(data) == 0:
    return []
  data = [r.to_dict() for r in data]

  data = filterByBodiesOfWater(data, bodiesOfWater)

  if orderedBy is not None:
    data = sort_images(data, orderedBy.lower(), isAscending, isDecending)

  return data[:limit]

def count_all():
  return images.count_all()

def count_by_submitter(contributors_email):
  return images.count_by_submitter(contributors_email)

def toTimestamp(d):
  t = d['timestamp']
  if t is None:
    t = d['createdTimestamp']
  if '.' in t:
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ")
  else:
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")

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
    for r in latLong:

      # todo: double check if this wrap-around calculation is correct
      maxLon = r.maxLongitude
      dataLon = d['longitude']
      if maxLon < r.minLongitude:
        maxLon = maxLon % 180 + 180
        if dataLon < 0:
          dataLon = dataLon % 180 + 180

      if r.minLatitude <= d['latitude'] <= r.maxLatitude and r.minLongitude <= dataLon <= maxLon:
        filtered_data.append(d)

  return filtered_data


properties = {
    "concept": {
        "type": "string",
        "description": "Specify a specific biological concept or category"
    },
    "contributorsEmail": {
        "type": "string",
        "description": "Contributors Email"
    },

    "includeUnverified": {
        "type": "boolean",
        "description": "Include verified species"
    },

    "includeVerified": {
        "type": "boolean",
        "description": "Include unverified species"
    },

    "limit": {
        "type": "number",
        "description": "Limit the number of images"
    },

    "maxDepth": {
        "type": "number",
        "description": "Maximum depth the species is found in"
    },
    "minDepth": {
        "type": "number",
        "description": "Minimum depth the species is found in"
    },
    "taxaProviderName": {
        "type": "number",
        "description": "The taxonomic provider name"
    },

    # sorting
    "orderedBy": {
        "type": "string",
        "description": "Sort the images"
    },
    "isDecending": {
        "type": "boolean",
        "description": "Sort in descending order"
    },
    "isAscending": {
        "type": "boolean",
        "description": "Sort in ascending order"
    },

    # location
    "bodiesOfWater": {
        "type": "string",
        "description": "The bodies of water the species is found in"
    },
    "latitude": {
        "type": "string",
        "description": "Latitude the species is found in"
    },
    "longitude": {
        "type": "string",
        "description": "Longitude the species is found in"
    },
    "kilometersFrom": {
        "type": "number",
        "description": "Kilometers away from location"
    },
    "milesFrom": {
        "type": "number",
        "description": "Miles away from location"
    },

    # quality
    "includeGood": {
        "type": "boolean",
        "description": "Include good images only"
    },
}

available_funcs = [
    {
        "name": "findImages",
        "description": "Get a constrained list of images.",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": []
        }
    },
    {
        "name": "count_all",
        "description": "Get a count of all images.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "count_by_submitter",
        "description": "Get a count of images by contributor.",
        "parameters": {
            "type": "object",
            "properties": {
                "contributors_email": {
                    "type": "string",
                    "description": "The contributor's email"
                }
            },
            "required": ["contributors_email"]
        }
    }
        ]

def call_function(function_call):
    function_name = function_call.get('name')
    args = json.loads(function_call.get('arguments'))

    for key, value in args.items():
        if isinstance(value, str):
            if value == "True":
                args[key] = True
            elif value == "False":
                args[key] = False
            else:
                try:
                    args[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    pass


    function_to_call = globals().get(function_name)
    if function_to_call:
        return function_to_call(**args)
    else:
        return f"No function named '{function_name}' in the global scope"

def get_response(messages):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        temperature=0,
        messages=messages,
        functions=available_funcs,
        function_call="auto",
    )
def run_promptv1(prompt, messages=[]):
    messages.append({"role": "user", "content": prompt})
    chatResponse = get_response(messages)
    messages.append(
        {"role": "assistant", "content": json.dumps(chatResponse.choices[0])},
    )
    if hasattr(chatResponse.choices[0].message, 'function_call'):
        function_response = call_function(chatResponse.choices[0].message.function_call)
        messages.append({"role": "function", "name": chatResponse.choices[0].message.function_call.name, "content": json.dumps(function_response)})
        messages.append({"role": "user", "content": "Summarize the last function content in a human readable format"})
        summaryResponse = get_response(messages)
        if(chatResponse.choices[0].message.function_call.name=='findImages'):
            return {'images': function_response, 'text': summaryResponse.choices[0].message.content}
        else:
            return {'text': summaryResponse.choices[0].message.content}
    else:
        return {'text': chatResponse.choices[0].message.content}



#print(json.dumps(run_prompt("Find me 3 newest images of species 'Aurelia aurita'")))
#print("----")
#print(json.dumps(run_prompt("Find me images of species 'Aurelia aurita' in Monterey Bay and depth less than 5000m", [])))
#print(json.dumps(run_prompt("Which image has the largest depth?")))

