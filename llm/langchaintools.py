from .constants import *
from .utils import getScientificNames

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
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

def getConceptCandidates(df, product_description, n=LANGCHAIN_SEARCH_CONCEPTS_TOPN, pprint=False):
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
    results = getScientificNames(description)
    candidates = getConceptCandidates(df, description)
    results.extend(candidates.values.tolist())
    results = list(dict.fromkeys(results))
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

    rows = cursor.fetchall()
    isJSON = False

    # Concatenate all rows to form a single string
    content = ''.join(str(row[0]) for row in rows)

    if content:
        try:
            # Try to load the content string as JSON
            decoded = json.loads(content)
            # Check if decoded is a JSON structure (dict or list)
            if isinstance(decoded, (dict, list)):
                output = decoded
                isJSON = True
            else:
                # If decoded is a basic data type, treat it as a table
                raise ValueError("Not a JSON structure")
        except (json.JSONDecodeError, ValueError):
            # Content is not JSON, treat it as a table
            if rows:
                columns = [column[0] for column in cursor.description]
                output = [dict(zip(columns, row)) for row in rows]
            else:
                output = None
    else:
        output = None

    return (isJSON, output)


def generateSQLQuery(
    prompt: str
) -> (bool, str):
    """Converts text to sql. It is important to provide a scientific name when a species data is provided. If the data is need for specific task, input the task too."""

    sql_generation_model = ChatOpenAI(model_name=SQL_FINE_TUNED_MODEL,temperature=0, openai_api_key = openai.api_key)

    sqlQuery = sql_generation_model.invoke([
        SystemMessage(content="You are a text-to-sql generator. You have a database of marine species, with marine regions, images, bounding boxes table. You must provide the response only in sql format. The sql should be generated in a way such that the response from sql is also in the expected format. ONLY return the sql query and nothing more."),
        HumanMessage(content="""
            The database has the following structure.

                """
                +DB_STRUCTURE+
                """
                If the prompt is asking about species or images of individual species, draft the sql in such a way that it generates json array containing the species data. Species data must contain species concept and bounding box id as id.

                Output only the sql query. Prompt: """ + prompt)
    ])

    print(sql_generation_model)
    
    return sqlQuery.content

    


def initLangchain():
    
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
    data = agent_chain(getSciNamesPrompt(concept))
    data = data['output']
    if DEBUG_LEVEL >= 1:
        print('Fetched scientific names from Langchain:')
        print(data)
    data = data.strip().split(', ')
    return data

def get_Response(prompt, agent_chain):
    sql_query = agent_chain("Your function is to generate sql for the prompt using the tools provided. Output only the sql query. Prompt: "+prompt)
    print(sql_query['output'])
    isJSON, result = GetSQLResult(sql_query['output'])


    summerizerResponse = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{"role": "system","content":"""You are a summarizer. You summarize the data, find out the outputType and output a json in the format. The response must be a json
        {
            "outputType": "", //enum(image, histogram, text, table)
            "summary": "", //string
        }

        The outputType should be based on the input and the summary should be based on the output
        """},{"role":"user", "content": "{\"input\": \"" + prompt + "\", \"output\":\"" + str(result)[:NUM_RESULTS_TO_SUMMARIZE] + "\"}"}],
    )

    summaryPromptResponse = json.loads(summerizerResponse["choices"][0]["message"]["content"])
    output = {
        "outputType": summaryPromptResponse["outputType"],
        "responseText": summaryPromptResponse["summary"]
    }
    if(isJSON):
        output["species"] = result
    else:
        output["table"] = result

    return output

    



agent_chain = initLangchain()

#DEBUG_LEVEL = 5
#print(agent_chain(getSciNamesPrompt('fused carapace'))['output'])

#print(getScientificNamesLangchain('rattail'))
#print(get_Response("Show the distribution of all species in Monterey Bay by depth using standard ocean depth levels", agent_chain))
print(get_Response("Generate a heatmap of species in Monterey Bay", agent_chain))
#print(get_Response("Show me images of Aurelia Aurata from Moneterey Bay", agent_chain))
#print(get_Response("Find me images of species 'Aurelia aurita' in Monterey bay and depth less than 5k meters", agent_chain))
#print(get_Response("What is the total number of images in the database?", agent_chain))
#print(get_Response("What is the the most found species in the database and what is it's location?", agent_chain))
