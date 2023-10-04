from .constants import *
from .utils import getScientificNames, isNameAvaliable, findDescendants, findAncestors, getParent, findRelatives, filterUnavailableDescendants

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
from langchain.agents import initialize_agent, AgentType, StructuredChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts.chat import ChatPromptTemplate

import os
os.environ["OPENAI_API_KEY"] = KEYS['openai']
openai.api_key = KEYS['openai']



# ==== Taxonomy ====

def getTaxonomyTree(
    scientificName: str
) -> list:
    """Function to get the taxonomy tree contain the names and taxonomic ranks of the ancestors and descendants for a scientific name. 
    ONLY return a machine-readable JSON object, and nothing more."""
    descendants = filterUnavailableDescendants(findDescendants(scientificName, species_only=False))
    ancestors = findAncestors(scientificName)
    
    rank = ""
    for d in descendants:
        if d.name.lower() == scientificName.lower():
            rank = d.rank
    
    descendants = [{'name': d.name, 'rank': d.rank.lower(), 'parent': getParent(d.name)} for d in descendants if d.name.lower() != scientificName.lower()]
    ancestors = [{'name': d.name, 'rank': d.rank.lower()} for d in ancestors]
    
    return json.dumps({'concept': scientificName, 'rank': rank.lower(), 'taxonomy': {'descendants': descendants, 'ancestors': ancestors}})


def getRelatives(
    scientificName: str
) -> list:
    """Function to get the closest taxonomic relatives for a scientific name."""
    relatives = findRelatives(scientificName)
    relatives = [d.name for d in relatives if d.name.lower() != scientificName.lower()]
    
    return json.dumps({'concept': scientificName, 'relatives': relatives})


# ==== Scientific name mapping ====

def getConceptCandidates(product_description, n=LANGCHAIN_SEARCH_CONCEPTS_TOPN, pprint=False):
    df = pd.read_csv(CONCEPTS_EMBEDDING)
    df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

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
    template = """A user will pass in a description, and you should select all objects from names that exactly fit the description.
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
    """Function to get all scientific names that fits a common name or appearance.
    DO NOT use this tool for descriptions of location, depth, taxonomy, salinity, or temperature"""
    if isNameAvaliable(description):
        return description
    results = getScientificNames(description)
    candidates = getConceptCandidates(description)
    results.extend(candidates.values.tolist())
    results = list(dict.fromkeys(results))
    print(results)
    results = filterScientificNames(description, results)
    print(results)
    return results


def getSciNamesPrompt(concept):
    template = """ONLY return a comma-separated list, and nothing more."""
    human_template = "Get me scientific names of "+concept+"."
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    return chat_prompt

def getScientificNamesLangchain(concept):
    agent_chain = initLangchain()
    data = agent_chain(getSciNamesPrompt(concept))
    data = data['output']
    if DEBUG_LEVEL >= 1:
        print('Fetched scientific names from Langchain:')
        print(data)
    data = data.strip().split(', ')
    return data
    

# ==== SQL database query ====

def GetSQLResult(query:str):
    isSpeciesData = False
    output = ""
    try:
        connection = pymssql.connect(
            server=KEYS['SQL_server'],
            user=KEYS['Db_user'],
            password=KEYS['Db_pwd'],
            database=KEYS['Db']
        )
        
        cursor = connection.cursor()
        
        cursor.execute(query)

        rows = cursor.fetchall()

        # Concatenate all rows to form a single string
        content = ''.join(str(row[0]) for row in rows)

        if content:
            try:
                # Try to load the content string as JSON
                decoded = json.loads(content)
                # Check if decoded is a JSON structure (dict or list)
                if isinstance(decoded, (dict, list)):
                    output = decoded
                    isSpeciesData = True
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
    except:
        print("Error processing sql server response")
        output = query

    return (isSpeciesData, output)


def generateSQLQuery(
    prompt: str
) -> (bool, str):
    """Converts text to sql. If the common name of a species is provided, it is important convert it to its scientific name. If the data is need for specific task, input the task too. The database has image, bounding box and marine regions table. The database has data of species in a marine region with the corresponding images."""

    sql_generation_model = ChatOpenAI(model_name=SQL_FINE_TUNED_MODEL,temperature=0, openai_api_key = openai.api_key)

    sqlQuery = sql_generation_model.invoke([
        SystemMessage(content="You are a text-to-sql generator. You have a database of marine species, with marine regions, images, bounding boxes table. You must provide the response only in sql format. The sql should be generated in a way such that the response from sql is also in the expected format. ONLY return the sql query and nothing more."),
        HumanMessage(content="""
            The database has the following structure.

                """
                +DB_STRUCTURE+
                """

                There is no direct mapping between marine region and image data, use the latitude/longitude data to search in a region.
                If the prompt is asking about species or images of individual species, draft the sql in such a way that it generates json array containing the species data. Species data must contain species concept and bounding box id as id.

                Output only the sql query. Prompt: """ + prompt)
    ])
    
    return sqlQuery.content


    
# ==== Bounding box processing ====

def getOtherCreaturesInImage(
    boundingBoxes: str
) -> list:
    """Function to find other species in each image. You must first query the database for bounding boxes. The input must be in the format of a machine-readable JSON list of bounding box data."""
    print(boundingBoxes)
    return [{'name': 'Aegina rosea', 'frequency': 4}, {'name': 'Aegina citrea', 'frequency': 2}]
    
def getImageQualityScore(
    images: str
) -> list:
    """Function to calculate the score for image quality, the higher the better. To get the best images, sort by this score. You must first query the database for images and bounding boxes. The input must be in the format of a machine-readable JSON string of image data containing bounding boxes."""
    print(image)
    return 0.5
    

# ==== Main Langchain functions ====

def genTool(function):
    return StructuredTool.from_function(function)

def initLangchain(messages=[]):
    
    memory = ConversationBufferMemory(memory_key="chat_history")

    for m in messages:
        memory.save_context({"input": m['prompt']}, {"output": m['response']})

        
    chat = ChatOpenAI(model_name="gpt-4-0613",temperature=0, openai_api_key = openai.api_key)
    tools = [
        genTool(getScientificNamesFromDescription), 
        genTool(generateSQLQuery),
        #genTool(GetSQLResult),
        genTool(getTaxonomyTree),
        genTool(getRelatives),
        #genTool(getOtherCreaturesInImage),
        #genTool(getImageQualityScore)
    ]


    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = StructuredChatAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    
    
    llm_chain = LLMChain(llm=chat, prompt=prompt)
    agent = StructuredChatAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )


# messages must be in the format: [{"prompt": prompt, "response": json.dumps(response)}]
def get_Response(prompt, messages=[]):
    formattedMessage = []
    curResponseFormat = {
        "prompt": "",
        "response": ""
    }
    for message in messages:
        if(message["role"] == 'user'):
            curResponseFormat["prompt"] = message["content"]
        elif(message["role"] == 'assistant'):
            curResponseFormat["response"] = message["content"]
            formattedMessage.append(curResponseFormat.copy())
            curResponseFormat["prompt"] = ""
            curResponseFormat["response"] = ""
    
    agent_chain = initLangchain(formattedMessage)
    
    if DEBUG_LEVEL >= 3:
        print(agent_chain)


    result = agent_chain.run(input="""Your function is to generate either sql or JSON for the prompt using the tools provided. You may also need to lookup the previous responses. Output only the sql query or the JSON string. 
    If the result is sql, output it directly without converting it to JSON.
    Otherwise, add each result as a separate element in a JSON list.

        [
            {}, // first result
            {}, // second result
        ]
    
    All outputs must be converted to a string.
        
    Prompt: """+prompt)


    isSpeciesData = False
    try:
        result = json.loads(result)
        if not isinstance(result, list):
            result = [result]
    except:
        isSpeciesData, result = GetSQLResult(result)

    print(messages+[{"role": "system","content":"""
                   
        Based on the below details output a json in provided format. The response must be a json.
        
        {
            "outputType": "", //enum(image, text, table, heatmap, vegaLite, taxonomy) The data type based on the 'input'
            "summary": "", //Summary of the data based on the 'output', If there are no results, output will be None
            "vegaSchema": { // Visualization grammar, Optional, Only need when the input asks for visualization except heatmap
            }
        }

        """},{"role":"user", "content": "{\"input\": \"" + prompt + "\", \"output\":\"" + str(result)[:NUM_RESULTS_TO_SUMMARIZE] + "\"}"}])
    summerizerResponse = openai.ChatCompletion.create(
        model="gpt-4-0613",
        temperature=0,
        messages=messages+[{"role": "system","content":"""
                   
        Based on the below details output a json in provided format. The response must be a json.
        
        {
            "outputType": "", //enum(image, text, table, heatmap, vegaLite, taxonomy) The data type based on the 'input' and previous response, use table when the data can be respresented as rows and column
            "summary": "", //Summary of the data based on the 'output', If there are no results, output will be None
            "vegaSchema": { // Visualization grammar, Optional, Only need when the input asks for visualization except heatmap
            }
        }

        """},{"role":"user", "content": "{\"input\": \"" + prompt + "\", \"output\":\"" + str(result)[:NUM_RESULTS_TO_SUMMARIZE] + "\"}"}],
    )

    summaryPromptResponse = json.loads(summerizerResponse["choices"][0]["message"]["content"])
    output = {
        "outputType": summaryPromptResponse["outputType"],
        "responseText": summaryPromptResponse["summary"],
        "vegaSchema": summaryPromptResponse["vegaSchema"],
    }
    if(isSpeciesData):
        computedTaxonomicConcepts = []#adding taxonomy data to only the first species in the array with a given concept.
        #if isinstance(result, dict) or isinstance(result, list):
        #    for specimen in result:
        #        if "concept" in specimen and isinstance(specimen["concept"], str) and len(specimen["concept"]) > 0 and specimen["concept"] not in computedTaxonomicConcepts:
        #            taxonomyResponse = json.loads(getTaxonomyTree(specimen["concept"]))
        #            specimen["rank"] = taxonomyResponse["rank"]
        #            specimen["taxonomy"] = taxonomyResponse["taxonomy"]
        #            computedTaxonomicConcepts.append(specimen["concept"])
        output["species"] = result
    elif(summaryPromptResponse["outputType"]=="taxonomy"):
        if(isinstance(result, list)):
            output["species"] = result
        else:
            output["species"] = [result]
        output["outputType"] = "species"
    elif(summaryPromptResponse["outputType"]!="vegaSchema"):
        output["table"] = result

    return output





#DEBUG_LEVEL = 5
#print(agent_chain(getSciNamesPrompt('fused carapace'))['output'])
#print(getScientificNamesLangchain('rattail'))

#print(get_Response("Display a bar chart illustrating the distribution of all species in Monterey Bay, categorized by ocean zones."))
#print(get_Response("Display a pie chart that correlates salinity levels with the distribution of Aurelia aurita categorizing salinity levels from 30 to 38 with each level of width 1"))
#print(get_Response("Generate a heatmap of 20 species in Monterey Bay"))
#print(get_Response("Show me images of Aurelia Aurita from Monterey Bay"))
#print(json.dumps(get_Response("Find me 3 images of moon jellyfish in Monterey bay and depth less than 5k meters")))
#print(get_Response("What is the total number of images of Startfish in the database?"))
#print(get_Response("What is the the most found species in the database and what is it's location?"))
#print(json.dumps(get_Response("Show me the taxonomy tree of Euryalidae and Aurelia aurita")))
#print(json.dumps(get_Response("Show me the taxonomy tree of Euryalidae")))
#print(json.dumps(get_Response("Find me 3 images of creatures in Monterey Bay")))
#print(json.dumps(get_Response("Find me 3 images of creatures with tentacles")))

#test_msgs = [{"prompt": 'Find me images of Moon jellyfish', "response": json.dumps({'a': '123', 'b': '456'})}, {"prompt": 'What do they look like', "response": json.dumps({'responseText': 'They are pink and translucent'})}]
#print(get_Response("Where can I find them", test_msgs))
#print(get_Response("What color are they", test_msgs))

#print(json.loads(getTaxonomyTree('Asteroidea')))
