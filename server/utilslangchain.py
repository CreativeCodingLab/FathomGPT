from constants import *

import openai
import json
from datetime import datetime
import math
from urllib.request import urlopen
from urllib.parse import quote
from typing import Optional
import pandas as pd
import numpy as np
from ast import literal_eval
from openai.embeddings_utils import get_embedding, cosine_similarity
import pymssql

from langchain.tools.base import StructuredTool
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import initialize_agent, AgentType

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts.chat import ChatPromptTemplate

import os
os.environ["OPENAI_API_KEY"] = KEYS['openai']
openai.api_key = KEYS['openai']


df = pd.read_csv(CONCEPTS_EMBEDDING)

def getAvailableConcepts(df, product_description, n=LANGCHAIN_SEARCH_CONCEPTS_TOPN, pprint=False):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    df = df.sort_values("similarity", ascending=False)
    df = df[df['similarity'] > 0.8] 
    
    print(df.head(n))

    results = (
        df
        .head(n)
        .concepts
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results
    
def filterScientificNames(
    description: Optional[str] = None,
    names: Optional[list] = None
) -> str:
    template = """A user will pass in a description, and you should select all objects from names that fit the description.
    ONLY return a comma separated list, and nothing more."""
    human_template = "{description} {names}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI(model_name="gpt-4-0613",temperature=0, openai_api_key = openai.api_key)
    data = chain.invoke({"description": description, "names": names})
    return data


def getScientificNamesFromDescription(
    description: str
) -> list:
    """Function to get all scientific names that fit a description"""
    results = getAvailableConcepts(df, description)
    results = results.values.tolist()
    print(results)
    results = filterScientificNames(description, results)
    print(results)
    return results

def GetSQLResult(query:str):
    connection = pymssql.connect(
        server=KEYS['SQL_server'],
        user=KEYS['Db_user'],
        password=KEYS['Db_pwd'],
        database=KEYS['Db']
    )
    
    cursor = connection.cursor()

    cursor.execute(query)
    
    row = cursor.fetchone()
    isJSON = False

    if row is not None:
        # Check if the row has a single column
        if len(row) == 1:
            content = str(row[0])
            try:
                decoded = json.loads(content)
                # Check if decoded is a JSON structure (dict or list)
                if isinstance(decoded, (dict, list)):
                    output = decoded
                else:
                    raise ValueError("Not a JSON structure")
                isJSON = True
            except (json.JSONDecodeError, ValueError):
                # Not a JSON, treat it as a table
                columns = [column[0] for column in cursor.description]
                data = [tuple(row)]
                data.extend(cursor.fetchall())
                output = [dict(zip(columns, row)) for row in data]
        else:
            # Multiple columns, treat it as a table
            columns = [column[0] for column in cursor.description]
            data = [tuple(row)]
            data.extend(cursor.fetchall())
            output = [dict(zip(columns, row)) for row in data]
    else:
        output = None

    return (isJSON, output)

def generateSQLQuery(
    prompt: str
) -> (bool, str):
    """Converts text to sql. It is important to provide a scientific name when a species data is provided."""

    sql_generation_model = ChatOpenAI(model_name=SQL_FINE_TUNED_MODEL,temperature=0, openai_api_key = openai.api_key)

    sqlQuery = sql_generation_model.invoke([
        SystemMessage(content="You are a text-to-sql generator. You have a database of marine species, with marine regions, images, bounding boxes table. You must provide the response only in sql format. The sql should be generated in a way such that the response from sql is also in the expected format"),
        HumanMessage(content="""
            The database has the following structure.

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

                If the prompt is asking about species or images of individual species, draft the sql in such a way that it generates json array containing the species data. Species data must contain species concept and bounding box id as id.

                Prompt: """ + prompt)
    ])
    return sqlQuery.content

    


def initLangchain():
    df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)
    
    getScientificNamesFromDescription_tool = StructuredTool.from_function(
        getScientificNamesFromDescription,
        )
    generateSQLQuery_tool = StructuredTool.from_function(
        generateSQLQuery
        )

    
        
    chat = ChatOpenAI(model_name="gpt-4",temperature=0, openai_api_key = openai.api_key)
    tools = [getScientificNamesFromDescription_tool, generateSQLQuery_tool]
    return initialize_agent(tools,
                               chat,
                               agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                               verbose=(DEBUG_LEVEL >= 1))
                               
def getSciNamesPrompt(concept):
    template = """ONLY return a comma-separated list, and nothing more."""
    human_template = "Get me scientific names of "+concept+"."
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    return chat_prompt

def getScientificNamesLangchain(concept):
    try:
        data = agent_chain(getSciNamesPrompt(concept))
        data = data['output']
        if DEBUG_LEVEL >= 1:
            print('Fetched scientific names from Langchain:')
            print(data)
        data = data.strip().split(', ')
        return data
    except:
        if DEBUG_LEVEL >= 1:
            print('failed to fetch Langchain scientific names')
        return []

def get_Response(prompt):
    sql_query = agent_chain("Your function is to generate sql for the prompt using the tools provided. Output only the sql query. Prompt: "+prompt)
    sql_result = GetSQLResult(sql_query["output"])

    summerizerModel = ChatOpenAI(model_name="gpt-4-0613",temperature=0, openai_api_key = openai.api_key)
    summaryPrompt = summerizerModel.invoke([
        SystemMessage(content="""You are a summarizer. You summarize the data, find out the outputType and output a json in the format. The response must be a json
        {
            "outputType": "", //enum(image, histogram, text, table)
            "summary": "", //string
        }

        The outputType should be based on the input and the summary should be based on the output
        """),
        HumanMessage(content="{\"input\": \"" + prompt + "\", \"output\":\"" + str(sql_result[1][:3000]) + "\"}"),
    ])

    summaryPromptResponse = json.loads(summaryPrompt.content)
    output = {
        "outputType": summaryPromptResponse["outputType"],
        "responseText": summaryPromptResponse["summary"]
    }
    if(sql_result[0]):
        output["species"] = sql_result[1]
    else:
        output["table"] = sql_result[1]

    return output

    



agent_chain = initLangchain()

#DEBUG_LEVEL = 5
#print(agent_chain(getSciNamesPrompt('fused carapace'))['output'])

#print(getScientificNamesLangchain('moon jellyfish'))
get_Response("Provide the data that correlates depth with the distribution of Aurelia aurita")