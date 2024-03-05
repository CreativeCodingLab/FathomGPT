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
NUM_RESULTS_TO_SUMMARIZE = 200
SCI_NAME_MATCH_SIMILARITY = 0.8

NAMES_JSON = 'data/names_normalized.json'
SEMANTIC_MATCHES_JSON = 'data/semantic_matches.json'
CONCEPTS_JSON = 'data/concepts.json'
CONCEPTS_EMBEDDING = "data/concepts_names_desc_embeddings.csv"

EARTH_RADIUS=6378137 # Earth’s radius, sphere
MILES_TO_METERS=1609.34

SQL_FINE_TUNED_MODEL = 'ft:gpt-3.5-turbo-1106:forbeslab::8q54GZRV'#'ft:gpt-3.5-turbo-1106:forbeslab::8q4F8LLn'#'ft:gpt-3.5-turbo-0613:forbeslab::822X8OkV'#'ft:gpt-3.5-turbo-1106:forbeslab::8ozsHpQ5'

# ft:gpt-3.5-turbo-1106:forbeslab::8q54GZRV last working model

SQL_IMAGE_SEARCH_FINE_TUNED_MODEL = 'ft:gpt-3.5-turbo-1106:forbeslab::8qkm3fbS'
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
CREATE TABLE "dbo"."bounding_box_comments" ( "id" bigint NOT NULL, "bounding_box_uuid" uniqueidentifier NULL, "created_timestamp" datetime2(6) NULL, "email" varchar(254) NULL, "last_updated_timestamp" datetime2(6) NULL, "text" varchar(2048) NULL, "uuid" uniqueidentifier NOT NULL, "alternate_concept" varchar(255) NULL, "flagged" bit NULL, CONSTRAINT "PK__bounding__3213E83F71625CCD" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."bounding_boxes" ("id" bigint NOT NULL,"concept" varchar(255) NULL,"created_timestamp" datetime2(6) NULL,"group_of" bit NULL,"height" int NULL,"last_updated_timestamp" datetime2(6) NULL,"observer" varchar(256) NULL,"occluded" bit NULL,"truncated" bit NULL,"uuid" uniqueidentifier NOT NULL,"verification_timestamp" datetimeoffset(6) NULL,"verified" bit NULL,"verifier" varchar(256) NULL,"width" int NULL,"x" int NULL,"y" int NULL,"image_id" bigint NULL,"alt_concept" varchar(255) NULL,"user_defined_key" varchar(56) NULL, "magnitude" decimal(18,5) NULL, CONSTRAINT "PK__bounding__3213E83F3E4C2D08" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE dbo.bounding_box_image_feature_vectors ( bounding_box_id bigint NOT NULL, vector_index int NULL, vector_value decimal(18,5) NULL, CONSTRAINT FK_bounding_box_image_feature_vectors_bounding_box_id FOREIGN KEY (bounding_box_id) REFERENCES dbo.bounding_boxes (id) ON DELETE CASCADE);
CREATE TABLE "dbo"."bounding_boxes_aud" ( "id" bigint NOT NULL, "rev" int NOT NULL, "revtype" smallint NULL, "concept" varchar(255) NULL, "created_timestamp" datetime2(6) NULL, "group_of" bit NULL, "height" int NULL, "last_updated_timestamp" datetime2(6) NULL, "observer" varchar(256) NULL, "occluded" bit NULL, "truncated" bit NULL, "uuid" uniqueidentifier NULL, "verification_timestamp" datetimeoffset(6) NULL, "verified" bit NULL, "verifier" varchar(256) NULL, "width" int NULL, "x" int NULL, "y" int NULL, "image_id" bigint NULL, "alt_concept" varchar(255) NULL, "user_defined_key" varchar(56) NULL, CONSTRAINT "PK__bounding__BE3894F99D30F28A" PRIMARY KEY CLUSTERED("id","rev")
ON [PRIMARY]);
CREATE TABLE "dbo"."darwin_cores" ( "id" bigint NOT NULL, "access_rights" varchar(1024) NULL, "basis_of_record" varchar(64) NULL, "bibliographic_citation" varchar(512) NULL, "collection_code" varchar(64) NULL, "collection_id" varchar(2048) NULL, "data_generalizations" varchar(512) NULL, "dataset_id" uniqueidentifier NULL, "dataset_name" varchar(255) NULL, "dynamic_properties"varchar(2048) NULL, "information_withheld" varchar(255) NULL, "institution_code" varchar(255) NULL, "institution_id" varchar(255) NULL, "license" varchar(2048) NULL, "modified" datetimeoffset(6) NULL, "owner_institution_code" varchar(255) NULL, "record_language" varchar(35) NULL, "record_references" varchar(2048) NULL, "record_type" varchar(32) NULL, "rights_holder" varchar(255) NULL, "uuid" uniqueidentifier NOT NULL, "image_set_upload_id" bigint NULL, CONSTRAINT "PK__darwin_c__3213E83F92DAE497" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."fathomnet_identities" ( "id" bigint NOT NULL, "api_key" varchar(255) NULL, "created_timestamp" datetime2(6) NULL, "disabled" bit NULL, "display_name" varchar(255) NULL, "email" varchar(255) NULL, "expertise_rank" varchar(32) NULL, "firebase_uid" varchar(255) NULL, "job_title" varchar(255) NULL, "last_updated_timestamp" datetime2(6) NULL, "organization" varchar(255) NULL, "profile" varchar(1024) NULL, "role_data" varchar(255) NULL, "uuid" uniqueidentifier NOT NULL, "avatar_url" varchar(2000) NULL, "orcid" varchar(32) NULL, "notification_frequency" varchar(32) NULL, CONSTRAINT "PK__fathomne__3213E83F59FE1468" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."fathomnet_identities_aud" ( "id" bigint NOT NULL, "rev" int NOT NULL, "revtype" smallint NULL, "api_key" varchar(255) NULL, "avatar_url" varchar(2000) NULL, "created_timestamp" datetime2(6) NULL, "disabled" bit NULL, "display_name" varchar(255) NULL, "email" varchar(254) NULL, "expertise_rank" varchar(32) NULL, "firebase_uid" varchar(64) NULL, "job_title" varchar(255) NULL, "last_updated_timestamp" datetime2(6) NULL, "orcid" varchar(32) NULL, "organization" varchar(255) NULL, "profile" varchar(1024) NULL, "role_data" varchar(255) NULL, "uuid" uniqueidentifier NULL, "notification_frequency" varchar(32) NULL, CONSTRAINT "PK__fathomne__BE3894F9CD98EF78" PRIMARY KEY CLUSTERED("id","rev")
ON [PRIMARY]);
CREATE TABLE "dbo"."followed_topics" ( "id" bigint NOT NULL, "created_timestamp" datetime2(6) NULL, "email" varchar(254) NULL, "last_updated_timestamp" datetime2(6) NULL, "notification" bit NULL, "target" varchar(256) NULL, "topic" varchar(32) NULL, "uuid" uniqueidentifier NOT NULL, CONSTRAINT "PK__followed__3213E83F4A1EA9E0" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."image_set_uploads" ( "id" bigint NOT NULL, "contributors_email" varchar(255) NULL, "created_timestamp" datetime2(6) NULL, "format" varchar(255) NULL, "last_updated_timestamp" datetime2(6) NULL, "local_path" varchar(2048) NULL, "rejection_details" varchar(255) NULL, "rejection_reason" varchar(255) NULL, "remote_uri" varchar(2048) NULL, "sha256" varchar(64) NULL, "status" varchar(255) NULL, "status_update_timestamp" datetimeoffset(6) NULL, "status_updater_email" varchar(254) NULL, "uuid" uniqueidentifier NOT NULL, "darwincore_id" bigint NULL, CONSTRAINT "PK__image_se__3213E83F9C72A0E9" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."image_uploads_join" ( "imagesetupload_id" bigint NOT NULL, "image_id" bigint NOT NULL, CONSTRAINT "PK__image_up__8A53EE0EBD9D776A" PRIMARY KEY CLUSTERED("imagesetupload_id","image_id")
ON [PRIMARY]);
CREATE TABLE "dbo"."images" ( "id" bigint NOT NULL, "created_timestamp" datetime2(6) NULL, "depth_meters" float NULL, "height" int NULL, "imaging_type" varchar(64) NULL, "last_updated_timestamp" datetime2(6) NULL, "last_validation" datetimeoffset(6) NULL, "latitude" float NULL, "longitude" float NULL, "media_type" varchar(255) NULL, "modified" datetimeoffset(6) NULL, "sha256" varchar(64) NULL, "submitter" varchar(255) NULL, "timestamp" datetimeoffset(6) NULL, "url" varchar(2048) NULL, "uuid" uniqueidentifier NOT NULL, "valid" bit NULL, "width" int NULL, "contributors_email"varchar(254) NULL, "altitude" float NULL, "oxygen_ml_l" float NULL, "pressure_dbar" float NULL, "salinity" float NULL, "temperature_celsius" float NULL, CONSTRAINT "PK__images__3213E83FD76E309F" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."images_aud" ( "id" bigint NOT NULL, "rev" int NOT NULL, "revtype" smallint NULL, "contributors_email"varchar(254) NULL, "created_timestamp" datetime2(6) NULL, "depth_meters" float NULL, "height" int NULL, "imaging_type" varchar(64) NULL, "last_updated_timestamp" datetime2(6) NULL, "last_validation" datetimeoffset(6) NULL, "latitude" float NULL, "longitude" float NULL, "media_type" varchar(255) NULL, "modified" datetimeoffset(6) NULL, "sha256" varchar(64) NULL, "timestamp" datetimeoffset(6) NULL, "url" varchar(255) NULL, "uuid" uniqueidentifier NULL, "valid" bit NULL, "width" int NULL, "altitude" float NULL, "oxygen_ml_l" float NULL, "pressure_dbar" float NULL, "salinity" float NULL, "temperature_celsius" float NULL, CONSTRAINT "PK__images_a__BE3894F92DF6715C" PRIMARY KEY CLUSTERED("id","rev")
ON [PRIMARY]);
CREATE TABLE "dbo"."marine_regions" ( "id" bigint NOT NULL, "created_timestamp" datetime2 NULL, "last_updated_timestamp" datetime2 NULL, "max_latitude" float NOT NULL, "max_longitude" float NOT NULL, "min_latitude" float NOT NULL, "min_longitude" float NOT NULL, "mrgid" bigint NULL, "name" varchar(255) NULL, CONSTRAINT "PK__marine_r__3213E83FE987FC8B" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."revinfo" ( "rev" int IDENTITY(1,1) NOT NULL, "revtstmp" bigint NULL, CONSTRAINT "PK__revinfo__C2B7CC69D3938648" PRIMARY KEY CLUSTERED("rev")
ON [PRIMARY]);
CREATE TABLE "dbo"."imageSourceInfo" ( "id" bigint NOT NULL, "created_timestamp" datetime2 NULL, "tag" varchar(255) NOT NULL, "last_updated_timestamp" datetime2 NULL, "media_type" varchar(255) NULL, "uuid" uniqueidentifier NOT NULL, "value" varchar(255) NOT NULL, "image_id" bigint NULL, CONSTRAINT "PK__imageSourceInfo__3213E83F3A490684" PRIMARY KEY CLUSTERED("id")
ON [PRIMARY]);
CREATE TABLE "dbo"."imageSourceInfo_aud" ( "id" bigint NOT NULL, "rev" int NOT NULL, "revtype" smallint NULL, "created_timestamp" datetime2 NULL, "tag" varchar(255) NULL, "last_updated_timestamp" datetime2 NULL, "media_type" varchar(255) NULL, "uuid" uniqueidentifier NULL, "value" varchar(255) NULL, "image_id" bigint NULL, CONSTRAINT "PK__imageSourceInfo_aud__BE3894F94F1608A8" PRIMARY KEY CLUSTERED("id","rev")
ON [PRIMARY]);
CREATE VIEW "dbo"."boundingbox_extended_info" AS
SELECT b.concept, b.alt_concept, b.observer, b.verified, b.verifier, b.verification_timestamp, b.user_defined_key, i.url, i.width, i.height, i.submitter, i.[timestamp], i.contributors_email AS image_contributors_email, u.contributors_email AS upload_contributors_email, dc.owner_institution_code, dc.institution_code, dc.rights_holder, dc.collection_code, dc.collection_id, dc.dataset_name
FROM dbo.darwin_cores dc LEFT JOIN dbo.image_set_uploads u ON dc.id = u.darwincore_id LEFT JOIN dbo.image_uploads_join j ON j.imagesetupload_id = u.id LEFT JOIN dbo.images i ON j.image_id = i.id LEFT JOIN dbo.bounding_boxes b ON b.image_id = i.id;

Sample imageSourceInfo table data
id	created_timestamp	tag	last_updated_timestamp	media_type	uuid	value	image_id
2256713	2021-09-29 21:15:46.7170000	source	2021-09-29 21:15:46.7170000	text/plain	E142AF1D-E85D-4821-A640-254BED3C8D64	MBARI/VARS	2256712
"""


FEW_SHOT_DATA = {
    "images": {
        "instructions": "The sql query must have bounding box id of the species, concept of the species and the image url of the species on all inputs. Important: You must include the id and concept of boding boxes on the response. There must be an output json.",
        "user": """
            User Prompt: "Provide me few images of Asterias rubens, Acanthaster planci, Linckia laevigata, Protoreaster nodosus, Pycnopodia helianthoides""",
        "assistant": """
        {
            "sqlServerQuery": "SELECT TOP 10     i.url AS url,     b.concept AS concept,     b.id as id,     b.image_id as image_id FROM      dbo.bounding_boxes AS b JOIN      dbo.images AS i ON b.image_id = i.id WHERE      b.concept IN ('Asterias rubens', 'Acanthaster planci', 'Linckia laevigata', 'Protoreaster nodosus', 'Pycnopodia helianthoides')",
            "responseText": "Few images of Asterias rubens, Acanthaster planci, Linckia laevigata, Protoreaster nodosus, Pycnopodia helianthoides are shown below."
        }""",
        "user2": """
            User Prompt: "Find me images of Aurelia aurita that looks good""",
        "assistant2": """
        {
            "sqlServerQuery": "SELECT TOP 10 i.url, b.concept, b.id, i.id as image_id FROM dbo.bounding_boxes AS b JOIN dbo.images AS i ON b.image_id = i.id WHERE b.concept = 'Aurelia aurita' ORDER BY (b.width * b.height) / (i.width * i.height) DESC",
            "responseText": "Here is an image of Aurelia aurita that you may find appealing."
        }""",
    },
    "text": {
        "instructions": "Make sure the response text is a templated string so that data can be formatted inside the text. There must be an output json.",
        "user": f"""
            User Prompt: ""How many images of Pycnopodia helianthoides are in the database"
            Output type: text""",
        "assistant": """{
                "sqlServerQuery": "SELECT COUNT(*) as TotalImages FROM dbo.bounding_boxes  WHERE concept = 'Pycnopodia helianthoides'",
                "responseText": "There are {TotalImages} images of Pycnopodia helianthoides in the database."
            }""",
        "user2": f"""
            User Prompt: ""In what pressure level is the species with bounding box id 2258729 living at"
            Output type: text""",
        "assistant2": """{
                "sqlServerQuery": "SELECT images.pressure_dbar, bounding_boxes.concept FROM dbo.bounding_boxes JOIN dbo.images ON bounding_boxes.image_id = images.id WHERE bounding_boxes.id = 2258739;",
                "responseText": "The species with bounding box id 2258729 us {concept}. It is living at pressure {pressure_dbar} dbar."
            }""",
    },
    "imagesWithInput": {
        "instructions": """
            You are a very intelligent json generated that can generate highly efficient sql queries. You will be given an input prompt for which you need to generated the JSON in a format given below, nothing else.
            The Generated SQL must be valid for Micorsoft sql server
            The JSON format and the attributes on the JSON are provided below
            {
            "similarImageIDs": [],
            "similarBoundingBoxIDs": [],
            "similarImageSearch": true/false,
            "sqlServerQuery": "",
            "responseText": ""
            }
            similarImageIDs: these are the image id that will be provided by the user in the prompt on which image search needs to be done
            similarBoundingBoxIDs: these are the bounding_boxes id that will be provided by the user in the prompt on which bounding boxes search needs to be done
            similarImageSearch: this is a boolean field, that is true when the prompt says to find similar images, else it is false
            sqlServerQuery: This is the sql server query you need to generate based on the user's prompt. The database structure provided will be very useful to generate the sql query. 
            responseText: Suppose you are answering the user with the output from the prompt. You need to write the message in this section. When the response is text, you need to output the textResponse in a way the values from the generated sql can be formatted in the text

            The prompt will asks for similar images, there is another system that takes in the similarImageIDs and similarBoundingBoxIDs that you generated above to calculate the similarity search. You will suppose the similarity search is already done and you have sql table SimilaritySearch that has the input bounding box id as bb1, output bounding box id as bb2 and Cosine Similarity Score as CosineSimilarity. You will use this table and add the conditions that is given provided by the user. You will also ouput the ouput bounding box image url and the concept. The result must be ordered in descending order using the CosineSimilarity value. Also, you will take 10 top results unless specified by the prompt
            """+f"SQL Server Database Structure: ${DB_STRUCTURE}",
        "user": f"""
            User Prompt: Find me similar images of species that are not Bathochordaeus stygius
            """,
        "assistant": """"
        {
            "similarImageIDs": [],
            "similarBoundingBoxIDs": [],
            "similarImageSearch": true,
            "sqlServerQuery": "SELECT TOP 10     SS.bb1,     SS.bb2,     BB.concept,     IMG.url,     SS.CosineSimilarity FROM SimilaritySearch SS INNER JOIN bounding_boxes BB ON BB.id = SS.bb2 INNER JOIN images IMG ON BB.image_id = IMG.id WHERE BB.concept <> 'Bathochordaeus stygius' ORDER BY SS.CosineSimilarity DESC;",
            "responseText": "Here are the similar images of species that are not Bathochordaeus stygius."
        }""",
        "user2": f"""
            User Prompt: "Find me images of species that looks alike that live at oxygen level between 0.5 to 1 ml per liter"
            """,
        "assistant2": """"
        {
            "similarImageIDs": [],
            "similarBoundingBoxIDs": [],
            "similarImageSearch": true,
            "sqlServerQuery": "SELECT TOP 10     SS.bb1,     SS.bb2,     BB.concept,     IMG.url,     SS.CosineSimilarity FROM SimilaritySearch SS INNER JOIN bounding_boxes BB ON BB.id = SS.bb2 INNER JOIN images IMG ON BB.image_id = IMG.id WHERE IMG.oxygen_ml_l BETWEEN 0.5 AND 1 ORDER BY SS.CosineSimilarity DESC;",
            "responseText": "Below are images of species that looks alike that live at oxygen level between 0.5 to 1 ml per liter."
        }""",
    },
    "visualization": {
        "instructions": """
            You are a very intelligent json generated that can generate highly efficient sql queries. You will be given an input prompt for which you need to generated the JSON in a format given below, nothing else.
            You are the doing the first part of the visualization work. Once you generate the sql query, the user will generate plotly code to visualzie the data.
                The Generated SQL must be valid
                The JSON should have sampleData, sqlServerQuery and responseText attribute

            Donot do any data processing in the sqlServerQuery, it will later be done by user using plotly python library. sqlServerQuery must not have any GROUP BY clause in the sql query. Make the plotlycode complex hard. Do all data processing in the plotly code.

            sampleData: This is the sample data that you think is needed for the visualization. Sample data must not have attributes other than any column in the sql server database tables
            sqlServerQuery: This is the sql server query you need to generate based on the user's prompt. The database structure provided will be very useful to generate the sql query. The output from this sql query must match the structure of sampleData above
            responseText: Suppose you are answering the user with the output from the prompt. You need to write the message in this section. When the response is text, you need to output the textResponse in a way the values from the generated sql can be formatted in the text
        
            Guarantee that the produced SQL query is free of syntax errors and does not contain any comments.
            While generating the sql server query make sure the output when running the sql query, the output data format matches the sample format. Make sure the variable names match
            Important: Donot generate the sql query wrong, if you are selecting a column, make sure the table is also referenced properly. There must be a valid sqlServerQuery in the output JSON.
            Make sure the sql query outputs data in format specified by the sample data, make sure the variable names match.
            """+f"SQL Server Database Structure: ${DB_STRUCTURE}",
        "user": f"""
            User Prompt: "Display a bar chart illustrating the distribution of all species in Monterey Bay, categorized by ocean depth levels."
            """,
        "assistant": """
        {
            "sampleData": "{'concept':['dolphin', 'shark'], 'depth_meters':[10, 20]}",
            "sqlServerQuery": "SELECT b.concept, i.depth_meters FROM dbo.bounding_boxes b JOIN dbo.images i ON b.image_id = i.id  INNER JOIN marine_regions MR ON i.latitude BETWEEN MR.min_latitude AND MR.max_latitude     AND i.longitude BETWEEN MR.min_longitude AND MR.max_longitude WHERE i.depth_meters IS NOT NULL AND MR.name='Monterey Bay';",
            "responseText": "Below is a bar chart illustrating the distribution of all species in Monterey Bay, categorized by ocean depth levels."
        }""",
        "sampleData": "{'concept':['dolphin', 'shark'], 'depth_meters':[10, 20]}",
        "plotlyCode": """import plotly.express as px
import pandas as pd

def drawVisualization(data):
    df = pd.DataFrame(data)
    
    bins = [0, 200, 400, 600, 800, 1000]
    labels = ['0-200m', '200-400m', '400-600m', '600-800m', '800-1000m']
    df['Depth Zone'] = pd.cut(df['depth_meters'], bins=bins, labels=labels, right=False)
    
    # Aggregate data
    zone_species_count = df.groupby(['Depth Zone', 'concept']).size().reset_index(name='Count')
    
    # Plot
    fig = px.bar(zone_species_count, x='Depth Zone', y='Count', color='concept', title='Species Distribution by Depth Zone in Monterey Bay')
    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})

	return fig""",
        "user2": f"""
            User Prompt: "Generate an Interactive Time-lapse Map of Marine Species Observations Grouped by Year"
            """,
        "assistant2": """
        {
            "sampleData": "{     'concept': ['Species A', 'Species B', 'Species A', 'Species B'],     'latitude': [35.6895, 35.6895, 35.6895, 35.6895],     'longitude': [139.6917, 139.6917, 139.6917, 139.6917],  'ObservationYear': [2020, 2021, 2022, 2023]  }",
            "sqlServerQuery": "SELECT      bb.concept,     i.latitude,      i.longitude,      YEAR(i.timestamp) AS ObservationYear FROM      dbo.bounding_boxes bb JOIN dbo.images i ON bb.image_id = i.id WHERE      i.latitude IS NOT NULL AND      i.longitude IS NOT NULL AND     i.timestamp IS NOT NULL ORDER BY      i.timestamp;",
            "responseText": "Below is an Interactive Time-lapse Map of Marine Species Observations Grouped by Year."
        }""",
        "sampleData2": "{     'concept': ['Species A', 'Species B', 'Species A', 'Species B'],     'latitude': [35.6895, 35.6895, 35.6895, 35.6895],     'longitude': [139.6917, 139.6917, 139.6917, 139.6917],  'ObservationYear': [2020, 2021, 2022, 2023]  }",
        "plotlyCode2": """import plotly.express as px
import pandas as pd

def drawVisualization(data):
    df = pd.DataFrame(data)

    fig = px.scatter_geo(df,
                         lat='latitude',
                         lon='longitude',
                         color='concept',
                         animation_frame='ObservationYear',
                         title='Interactive Time-lapse Map of Marine Species Observations Grouped by Year',
                         size_max=15,
                         projection="natural earth")

    fig.update_layout(geo=dict(showland=True, landcolor="rgb(217, 217, 217)"))

    return fig""",
    "user3": f"""
            User Prompt: "Display me a heatmap of all species in Pacific Ocean"
            """,
    "sampleData3": "{     'concept': ['Species A', 'Species B', 'Species A', 'Species B'],     'latitude': [35.6895, 35.6895, 35.6895, 35.6895],     'longitude': [139.6917, 139.6917, 139.6917, 139.6917] }",
    "plotlyCode3": """import pandas as pd
import plotly.express as px
def drawVisualization(data):
    df = pd.DataFrame(data)
    fig = px.density_mapbox(df, lat='latitude', lon='longitude', radius=10, center=dict(lat=10.652585, lon=-137.942181), zoom=10, mapbox_style="open-street-map", hover_data={'concept': True})  
    fig.update_layout(title='Heatmap of Species in Pacific Ocean')
    return fig"""
    },
    "table": {
        "instructions": "The response text can be templated so that it can hold the count of the data array from the sql query result. There must be an output json.",
        "user": f"""
            User Prompt: "List the species that are found in image with id 2256720""",
        "user2": f"""
            User Prompt: "What species are frequently found at 1000m depth?""",
        "assistant": """
        {
            "sqlServerQuery": "SELECT b.concept FROM dbo.bounding_boxes AS b JOIN dbo.images AS i ON b.image_id = i.id WHERE b.image_id = 2256720;",
            "responseText": "The table below lists all the species found in image with id 2256720."
        }""",
        "assistant2": """
        {
            "sqlServerQuery": "SELECT b.concept AS species, COUNT(*) AS frequency FROM dbo.bounding_boxes AS b JOIN dbo.images AS i ON b.image_id = i.id WHERE i.depth_meters = 1000 GROUP BY b.concept ORDER BY frequency DESC;",
            "responseText": "Table shows the frequently found species at 1000m depth and their count."
        }""",
    },
}