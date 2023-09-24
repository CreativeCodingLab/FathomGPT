from dotenv import load_dotenv
import os
import inspect


load_dotenv()
KEYS = {
    "openai": os.getenv("OPENAI_KEY"),
    "SQL_server": os.getenv("SQL_SERVER"),
    "Db": os.getenv("DATABASE"),
    "Db_user": os.getenv("DB_USER"),
    "Db_pwd": os.getenv("DB_PWD"),
}

DEFAULT_LIMIT = 5
RETRIEVAL_LIMIT = 1000000
DEFAULT_TAXA_PROVIDER = 'fathomnet'
GOOD_BOUNDING_BOX_MIN_SIZE = 0.2
GOOD_BOUNDING_BOX_MIN_MARGINS = 0.01
FIND_DESCENDENTS_DEFAULT = True # always include descendents by default
ONLY_USE_LANGCHAIN_NAME_MAPPING = False 
LANGCHAIN_SEARCH_CONCEPTS_TOPN = 10

NAMES_JSON = 'data/names_normalized.json'
CONCEPTS_JSON = 'data/concepts.json'
CONCEPTS_EMBEDDING = "data/concepts_names_embeddings.csv"

EARTH_RADIUS=6378137 # Earth’s radius, sphere
MILES_TO_METERS=1609.34

SQL_FINE_TUNED_MODEL = 'ft:gpt-3.5-turbo-0613:forbeslab::822X8OkV'

DEBUG_LEVEL = 2
# only print debugging messages if the initial caller is debug.py
python_call_stack = inspect.stack()
if len(python_call_stack) == 0 or not python_call_stack[-1].filename.endswith('debug.py'):
    DEBUG_LEVEL = 0

FUNCTION_PROPERTIES = {
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

    # taxonomy
    "findChildren": {
        "type": "boolean",
        "description": "Find descendants or children of the species"
    },
    "findSpeciesBelongingToTaxonomy": {
        "type": "boolean",
        "description": "Find all species belonging to a taxonomic level"
    },
    "findParent": {
        "type": "boolean",
        "description": "Find the ancestor or parent of the species"
    },
    "findClosestRelative": {
        "type": "boolean",
        "description": "Find the closest relative of the species"
    },
    "taxaProviderName": {
        "type": "number",
        "description": "The taxonomic provider name"
    },

    # image
    "includeGood": {
        "type": "boolean",
        "description": "Include good images only"
    },
    "findBest": {
        "type": "boolean",
        "description": "Sort the images by highest quality"
    },
    "findWorst": {
        "type": "boolean",
        "description": "Sort the images by lowest quality"
    },
    "findOtherSpecies": {
        "type": "boolean",
        "description": "Find other species commonly found in the same image"
    },
    "excludeOtherSpecies": {
        "type": "boolean",
        "description": "Find images only containing the species, not containing other species"
    },
}

AVAILABLE_FUNCTIONS = [
    {
        "name": "findImages",
        "description": "Get a constrained list of images.",
        "parameters": {
            "type": "object",
            "properties": FUNCTION_PROPERTIES,
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
