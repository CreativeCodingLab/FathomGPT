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
RETRIEVAL_LIMIT = 1000
DEFAULT_TAXA_PROVIDER = 'fathomnet'
GOOD_BOUNDING_BOX_MIN_SIZE = 0.2
GOOD_BOUNDING_BOX_MIN_MARGINS = 0.01
FIND_DESCENDENTS_DEFAULT = True # always include descendents by default
LANGCHAIN_SEARCH_CONCEPTS_TOPN = 10
LLM_TYPE = 'langchain'
NUM_RESULTS_TO_SUMMARIZE = 1000
SCI_NAME_MATCH_SIMILARITY = 0.8

NAMES_JSON = 'data/names_normalized.json'
SEMANTIC_MATCHES_JSON = 'data/semantic_matches.json'
CONCEPTS_JSON = 'data/concepts.json'
CONCEPTS_EMBEDDING = "data/concepts_names_desc_embeddings.csv"

EARTH_RADIUS=6378137 # Earth’s radius, sphere
MILES_TO_METERS=1609.34

SQL_FINE_TUNED_MODEL = 'ft:gpt-3.5-turbo-0613:forbeslab::822X8OkV'

#'ft:gpt-3.5-turbo-0613:forbeslab::822X8OkV' Without observation gpt-3.5-turbo-0613
#'ft:gpt-3.5-turbo-0613:forbeslab::83TTzs4O' With observation gpt-3.5-turbo-0613

DEBUG_LEVEL = 0
# only print debugging messages if the initial caller is debug.py
def hideDebugMessagesInProd():
    stack = inspect.stack()
    if len(stack) == 0:
        DEBUG_LEVEL = 0
        return
    file = stack[-1].filename
    if not file.endswith('debug.py') and not file.endswith('langchaintools.py'):
        DEBUG_LEVEL = 0
hideDebugMessagesInProd()

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

DB_STRUCTURE="""
                CREATE TABLE "dbo"."bounding_box_comments"  (
                    "id"                    	bigint NOT NULL,
                    "bounding_box_uuid"     	uniqueidentifier NULL,
                    "created_timestamp"     	datetime2(6) NULL,
                    "email"                 	varchar(254) NULL,
                    "last_updated_timestamp"	datetime2(6) NULL,
                    "text"                  	varchar(2048) NULL,
                    "uuid"                  	uniqueidentifier NOT NULL,
                    "alternate_concept"     	varchar(255) NULL,
                    "flagged"               	bit NULL,
                    CONSTRAINT "PK__bounding__3213E83F71625CCD" PRIMARY KEY CLUSTERED("id")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."bounding_boxes"  (
                    "id"                    	bigint NOT NULL,
                    "concept"               	varchar(255) NULL,
                    "created_timestamp"     	datetime2(6) NULL,
                    "group_of"              	bit NULL,
                    "height"                	int NULL,
                    "last_updated_timestamp"	datetime2(6) NULL,
                    "observer"              	varchar(256) NULL,
                    "occluded"              	bit NULL,
                    "truncated"             	bit NULL,
                    "uuid"                  	uniqueidentifier NOT NULL,
                    "verification_timestamp"	datetimeoffset(6) NULL,
                    "verified"              	bit NULL,
                    "verifier"              	varchar(256) NULL,
                    "width"                 	int NULL,
                    "x"                     	int NULL,
                    "y"                     	int NULL,
                    "image_id"              	bigint NULL,
                    "alt_concept"           	varchar(255) NULL,
                    "user_defined_key"      	varchar(56) NULL,
                    CONSTRAINT "PK__bounding__3213E83F3E4C2D08" PRIMARY KEY CLUSTERED("id")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."bounding_boxes_aud"  (
                    "id"                    	bigint NOT NULL,
                    "rev"                   	int NOT NULL,
                    "revtype"               	smallint NULL,
                    "concept"               	varchar(255) NULL,
                    "created_timestamp"     	datetime2(6) NULL,
                    "group_of"              	bit NULL,
                    "height"                	int NULL,
                    "last_updated_timestamp"	datetime2(6) NULL,
                    "observer"              	varchar(256) NULL,
                    "occluded"              	bit NULL,
                    "truncated"             	bit NULL,
                    "uuid"                  	uniqueidentifier NULL,
                    "verification_timestamp"	datetimeoffset(6) NULL,
                    "verified"              	bit NULL,
                    "verifier"              	varchar(256) NULL,
                    "width"                 	int NULL,
                    "x"                     	int NULL,
                    "y"                     	int NULL,
                    "image_id"              	bigint NULL,
                    "alt_concept"           	varchar(255) NULL,
                    "user_defined_key"      	varchar(56) NULL,
                    CONSTRAINT "PK__bounding__BE3894F99D30F28A" PRIMARY KEY CLUSTERED("id","rev")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."darwin_cores"  (
                    "id"                    	bigint NOT NULL,
                    "access_rights"         	varchar(1024) NULL,
                    "basis_of_record"       	varchar(64) NULL,
                    "bibliographic_citation"	varchar(512) NULL,
                    "collection_code"       	varchar(64) NULL,
                    "collection_id"         	varchar(2048) NULL,
                    "data_generalizations"  	varchar(512) NULL,
                    "dataset_id"            	uniqueidentifier NULL,
                    "dataset_name"          	varchar(255) NULL,
                    "dynamic_properties"    	varchar(2048) NULL,
                    "information_withheld"  	varchar(255) NULL,
                    "institution_code"      	varchar(255) NULL,
                    "institution_id"        	varchar(255) NULL,
                    "license"               	varchar(2048) NULL,
                    "modified"              	datetimeoffset(6) NULL,
                    "owner_institution_code"	varchar(255) NULL,
                    "record_language"       	varchar(35) NULL,
                    "record_references"     	varchar(2048) NULL,
                    "record_type"           	varchar(32) NULL,
                    "rights_holder"         	varchar(255) NULL,
                    "uuid"                  	uniqueidentifier NOT NULL,
                    "image_set_upload_id"   	bigint NULL,
                    CONSTRAINT "PK__darwin_c__3213E83F92DAE497" PRIMARY KEY CLUSTERED("id")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."fathomnet_identities"  (
                    "id"                    	bigint NOT NULL,
                    "api_key"               	varchar(255) NULL,
                    "created_timestamp"     	datetime2(6) NULL,
                    "disabled"              	bit NULL,
                    "display_name"          	varchar(255) NULL,
                    "email"                 	varchar(255) NULL,
                    "expertise_rank"        	varchar(32) NULL,
                    "firebase_uid"          	varchar(255) NULL,
                    "job_title"             	varchar(255) NULL,
                    "last_updated_timestamp"	datetime2(6) NULL,
                    "organization"          	varchar(255) NULL,
                    "profile"               	varchar(1024) NULL,
                    "role_data"             	varchar(255) NULL,
                    "uuid"                  	uniqueidentifier NOT NULL,
                    "avatar_url"            	varchar(2000) NULL,
                    "orcid"                 	varchar(32) NULL,
                    "notification_frequency"	varchar(32) NULL,
                    CONSTRAINT "PK__fathomne__3213E83F59FE1468" PRIMARY KEY CLUSTERED("id")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."fathomnet_identities_aud"  (
                    "id"                    	bigint NOT NULL,
                    "rev"                   	int NOT NULL,
                    "revtype"               	smallint NULL,
                    "api_key"               	varchar(255) NULL,
                    "avatar_url"            	varchar(2000) NULL,
                    "created_timestamp"     	datetime2(6) NULL,
                    "disabled"              	bit NULL,
                    "display_name"          	varchar(255) NULL,
                    "email"                 	varchar(254) NULL,
                    "expertise_rank"        	varchar(32) NULL,
                    "firebase_uid"          	varchar(64) NULL,
                    "job_title"             	varchar(255) NULL,
                    "last_updated_timestamp"	datetime2(6) NULL,
                    "orcid"                 	varchar(32) NULL,
                    "organization"          	varchar(255) NULL,
                    "profile"               	varchar(1024) NULL,
                    "role_data"             	varchar(255) NULL,
                    "uuid"                  	uniqueidentifier NULL,
                    "notification_frequency"	varchar(32) NULL,
                    CONSTRAINT "PK__fathomne__BE3894F9CD98EF78" PRIMARY KEY CLUSTERED("id","rev")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."followed_topics"  (
                    "id"                    	bigint NOT NULL,
                    "created_timestamp"     	datetime2(6) NULL,
                    "email"                 	varchar(254) NULL,
                    "last_updated_timestamp"	datetime2(6) NULL,
                    "notification"          	bit NULL,
                    "target"                	varchar(256) NULL,
                    "topic"                 	varchar(32) NULL,
                    "uuid"                  	uniqueidentifier NOT NULL,
                    CONSTRAINT "PK__followed__3213E83F4A1EA9E0" PRIMARY KEY CLUSTERED("id")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."image_set_uploads"  (
                    "id"                     	bigint NOT NULL,
                    "contributors_email"     	varchar(255) NULL,
                    "created_timestamp"      	datetime2(6) NULL,
                    "format"                 	varchar(255) NULL,
                    "last_updated_timestamp" 	datetime2(6) NULL,
                    "local_path"             	varchar(2048) NULL,
                    "rejection_details"      	varchar(255) NULL,
                    "rejection_reason"       	varchar(255) NULL,
                    "remote_uri"             	varchar(2048) NULL,
                    "sha256"                 	varchar(64) NULL,
                    "status"                 	varchar(255) NULL,
                    "status_update_timestamp"	datetimeoffset(6) NULL,
                    "status_updater_email"   	varchar(254) NULL,
                    "uuid"                   	uniqueidentifier NOT NULL,
                    "darwincore_id"          	bigint NULL,
                    CONSTRAINT "PK__image_se__3213E83F9C72A0E9" PRIMARY KEY CLUSTERED("id")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."image_uploads_join"  (
                    "imagesetupload_id"	bigint NOT NULL,
                    "image_id"         	bigint NOT NULL,
                    CONSTRAINT "PK__image_up__8A53EE0EBD9D776A" PRIMARY KEY CLUSTERED("imagesetupload_id","image_id")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."images"  (
                    "id"                    	bigint NOT NULL,
                    "created_timestamp"     	datetime2(6) NULL,
                    "depth_meters"          	float NULL,
                    "height"                	int NULL,
                    "imaging_type"          	varchar(64) NULL,
                    "last_updated_timestamp"	datetime2(6) NULL,
                    "last_validation"       	datetimeoffset(6) NULL,
                    "latitude"              	float NULL,
                    "longitude"             	float NULL,
                    "media_type"            	varchar(255) NULL,
                    "modified"              	datetimeoffset(6) NULL,
                    "sha256"                	varchar(64) NULL,
                    "submitter"             	varchar(255) NULL,
                    "timestamp"             	datetimeoffset(6) NULL,
                    "url"                   	varchar(2048) NULL,
                    "uuid"                  	uniqueidentifier NOT NULL,
                    "valid"                 	bit NULL,
                    "width"                 	int NULL,
                    "contributors_email"    	varchar(254) NULL,
                    "altitude"              	float NULL,
                    "oxygen_ml_l"           	float NULL,
                    "pressure_dbar"         	float NULL,
                    "salinity"              	float NULL,
                    "temperature_celsius"   	float NULL,
                    CONSTRAINT "PK__images__3213E83FD76E309F" PRIMARY KEY CLUSTERED("id")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."images_aud"  (
                    "id"                    	bigint NOT NULL,
                    "rev"                   	int NOT NULL,
                    "revtype"               	smallint NULL,
                    "contributors_email"    	varchar(254) NULL,
                    "created_timestamp"     	datetime2(6) NULL,
                    "depth_meters"          	float NULL,
                    "height"                	int NULL,
                    "imaging_type"          	varchar(64) NULL,
                    "last_updated_timestamp"	datetime2(6) NULL,
                    "last_validation"       	datetimeoffset(6) NULL,
                    "latitude"              	float NULL,
                    "longitude"             	float NULL,
                    "media_type"            	varchar(255) NULL,
                    "modified"              	datetimeoffset(6) NULL,
                    "sha256"                	varchar(64) NULL,
                    "timestamp"             	datetimeoffset(6) NULL,
                    "url"                   	varchar(255) NULL,
                    "uuid"                  	uniqueidentifier NULL,
                    "valid"                 	bit NULL,
                    "width"                 	int NULL,
                    "altitude"              	float NULL,
                    "oxygen_ml_l"           	float NULL,
                    "pressure_dbar"         	float NULL,
                    "salinity"              	float NULL,
                    "temperature_celsius"   	float NULL,
                    CONSTRAINT "PK__images_a__BE3894F92DF6715C" PRIMARY KEY CLUSTERED("id","rev")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."marine_regions"  (
                    "id"                    	bigint NOT NULL,
                    "created_timestamp"     	datetime2 NULL,
                    "last_updated_timestamp"	datetime2 NULL,
                    "max_latitude"          	float NOT NULL,
                    "max_longitude"         	float NOT NULL,
                    "min_latitude"          	float NOT NULL,
                    "min_longitude"         	float NOT NULL,
                    "mrgid"                 	bigint NULL,
                    "name"                  	varchar(255) NULL,
                    CONSTRAINT "PK__marine_r__3213E83FE987FC8B" PRIMARY KEY CLUSTERED("id")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."revinfo"  (
                    "rev"     	int IDENTITY(1,1) NOT NULL,
                    "revtstmp"	bigint NULL,
                    CONSTRAINT "PK__revinfo__C2B7CC69D3938648" PRIMARY KEY CLUSTERED("rev")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."tags"  (
                    "id"                    	bigint NOT NULL,
                    "created_timestamp"     	datetime2 NULL,
                    "tag"                   	varchar(255) NOT NULL,
                    "last_updated_timestamp"	datetime2 NULL,
                    "media_type"            	varchar(255) NULL,
                    "uuid"                  	uniqueidentifier NOT NULL,
                    "value"                 	varchar(255) NOT NULL,
                    "image_id"              	bigint NULL,
                    CONSTRAINT "PK__tags__3213E83F3A490684" PRIMARY KEY CLUSTERED("id")
                ON [PRIMARY]);
                CREATE TABLE "dbo"."tags_aud"  (
                    "id"                    	bigint NOT NULL,
                    "rev"                   	int NOT NULL,
                    "revtype"               	smallint NULL,
                    "created_timestamp"     	datetime2 NULL,
                    "tag"                   	varchar(255) NULL,
                    "last_updated_timestamp"	datetime2 NULL,
                    "media_type"            	varchar(255) NULL,
                    "uuid"                  	uniqueidentifier NULL,
                    "value"                 	varchar(255) NULL,
                    "image_id"              	bigint NULL,
                    CONSTRAINT "PK__tags_aud__BE3894F94F1608A8" PRIMARY KEY CLUSTERED("id","rev")
                ON [PRIMARY]);
                CREATE VIEW "dbo"."boundingbox_extended_info" AS
                SELECT
                    b.concept,
                    b.alt_concept,
                    b.observer,
                    b.verified,
                    b.verifier,
                    b.verification_timestamp,
                    b.user_defined_key,
                    i.url,
                    i.width,
                    i.height,
                    i.submitter,
                    i.[timestamp],
                    i.contributors_email AS image_contributors_email,
                    u.contributors_email AS upload_contributors_email,
                    dc.owner_institution_code,
                    dc.institution_code,
                    dc.rights_holder,
                    dc.collection_code,
                    dc.collection_id,
                    dc.dataset_name
                FROM
                    dbo.darwin_cores dc LEFT JOIN
                    dbo.image_set_uploads u ON dc.id = u.darwincore_id LEFT JOIN
                    dbo.image_uploads_join j ON j.imagesetupload_id = u.id LEFT JOIN
                    dbo.images i ON j.image_id = i.id LEFT JOIN
                    dbo.bounding_boxes b ON b.image_id = i.id
                ;
"""