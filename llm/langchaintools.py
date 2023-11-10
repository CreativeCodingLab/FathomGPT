from .constants import *
from .utils import getScientificNames, isNameAvaliable, findDescendants, findAncestors, getParent, findRelatives, filterUnavailableDescendants, changeNumberToFetch, postprocess

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
import re
import time

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
    ancestors.reverse()
    
    return json.dumps({'concept': scientificName, 'rank': rank.lower(), 'taxonomy': {'descendants': descendants, 'ancestors': ancestors}})


def getTaxonomicRelatives(
    scientificName: str
) -> list:
    """Function to get the taxonomic relatives for a scientific name."""
    relatives = findRelatives(scientificName)
    relatives = [d.name for d in relatives if d.name.lower() != scientificName.lower()]
    
    return json.dumps({'concept': scientificName, 'relatives': relatives})

def getRelativesString(results):
    out = ''
    for result in results:
        relatives = result['relatives']
        scientificName = result['concept']
        if len(relatives) == 0:
            out = out + 'There are no relatives of '+scientificName+'. '
        elif len(relatives) == 1:
            out = out + 'The closest relative of '+scientificName+' is: '+ relatives[0]+'. '
        else:
            out = out + 'The relatives of '+scientificName+' are: '+ ', '.join(relatives)+'. '
    return out

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
    return data.content

def getScientificNamesFromDescription(
    name: str,
    constraints: str,
    description: str,
) -> list:
    """Function to get all scientific names that fits a common name or appearance.
    DO NOT use this tool for descriptions of location, depth, taxonomy, salinity, or temperature"""
    print("name: "+name+", description: "+description)
    
    results = []

    if len(name) > 0:
        if isNameAvaliable(name):
            return name
        results.extend(getScientificNames(name))

        if len(description) == 0:
            description = name
            
    desc = list(set(name.split(' ') + description.split(' ')))
    if len(description) > 0:
        desc.append(description)
    desc = list(set([d for d in desc if len(d)>0]))
        
    print(desc)
    for d in desc:
        results.extend(getScientificNames(d, False, SEMANTIC_MATCHES_JSON, True))
    
    if len(description) > 0 and len(results) == 0:
        results = list(getConceptCandidates(description))
    
    if len(results) > 0:
        if len(results) > LANGCHAIN_SEARCH_CONCEPTS_TOPN:
            results = results[:LANGCHAIN_SEARCH_CONCEPTS_TOPN]
        results = list(dict.fromkeys(results))
        print(results)
        return ", ".join(results)
    
    return "anything"


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
    prompt: str,
    scientificNames: str,
    name: str,
) -> str:
    """Converts text to sql. If the common name of a species is provided, it is important convert it to its scientific name. If the data is need for specific task, input the task too. The database has image, bounding box and marine regions table. The database has data of species in a marine region with the corresponding images.
    """
    
    if len(name) > 1:
        prompt = prompt.replace(name, scientificNames)
    elif len(scientificNames) > 1:
        prompt = prompt.rstrip(".")+" with names: "+scientificNames
    print(prompt)

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
                If the prompt is asking about multiple species draft the sql to query for a list of species.
                If the prompt is asking about creatures found in the same image as a species, return sql to find images of the species instead.

                Output only the sql query. Prompt: """ + prompt)
    ])
    
    return sqlQuery.content



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
        genTool(getTaxonomicRelatives),
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

availableFunctions = [{
    "name": "getScientificNamesFromDescription",
    "description": "Function to get all scientific names that fits a common name or appearance. If there are no matches, return anything. DO NOT use this tool for descriptions of location, depth, taxonomy, salinity, or temperature",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the creature",
            },
            "constraints": {
                "type": "string",
                "description": "The location, depth, taxonomy, salinity, or temperature",
            },
            "description": {
                "type": "string",
                "description": "The description of the species, excluding location, depth, taxonomy, salinity, or temperature",
            },
        },
        "required": ["name", "constraints", "description"],
    },
},{
    "name": "generateSQLQuery",
    "description": "Converts text to sql. If no scientific name of a species is provided, it is important convert it to its scientific name. If the data is need for specific task, input the task too. The database has image, bounding box and marine regions table. The database has data of species in a marine region with the corresponding images.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text prompt that can be used to generate the sql query. The prompt cannot have common name of any species",
            },
            "scientificNames": {
                "type": "string",
                "description": "The scientific names of all of the species",
            },
            "name": {
                "type": "string",
                "description": "The name of the species from the prompt",
            },
        },
        "required": ["prompt", "scientificNames", "name"],
    },
},{
    "name": "getTaxonomyTree",
    "description": "Function to get the taxonomy tree contain the names and taxonomic ranks of the ancestors and descendants for a scientific name. ONLY return a machine-readable JSON object, and nothing more.",
    "parameters": {
        "type": "object",
        "properties": {
            "scientificName": {
                "type": "string",
                "description": "Scientific name of the species",
            },
        },
        "required": ["scientificName"],
    },
},{
    "name": "getTaxonomicRelatives",
    "description": "Function to get the taxonomic relatives for a scientific name.",
    "parameters": {
        "type": "object",
        "properties": {
            "scientificName": {
                "type": "string",
                "description": "Scientific name of the species who taxonomic relative is to be found out",
            },
        },
        "required": ["scientificName"],
    },
}
]

availableFunctionsDescription = {
    "getScientificNamesFromDescription": "Generating scientific name from description",
    
    "generateSQLQuery": "Generating SQL Query",
    
    "getTaxonomyTree": "Getting the taxonomy tree",
    
    "getTaxonomicRelatives": "Getting taxonomic relatives",
}

# messages must be in the format: [{"prompt": prompt, "response": json.dumps(response)}]
def get_Response(prompt, messages=[], isEventStream=False):
    start_time = time.time()
    modifiedMessages = []
    for smessage in modifiedMessages:
        if(smessage["role"]=="assistant"):
            if(len(smessage["content"])>200):
                modifiedMessages["content"]=smessage["content"][:200]+"...\n"
    modifiedMessages.append({"role":"user","content":"Use the tools provided to generate response to the prompt. Important: If the prompt contains a common name or description use the 'getScientificNamesFromDescription' tool first. Prompt:"+prompt})
    isSpeciesData = False
    result = None
    curLoopCount = 0

    if isEventStream:
        event_data = {
            "message": "Evaluating Prompt"
        }
        sse_data = f"data: {json.dumps(event_data)}\n\n"
        yield sse_data

    while curLoopCount < 4:
        curLoopCount+=1
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=modifiedMessages,
            functions=availableFunctions,
            function_call="auto",
            temperature=0,

        )
        response_message = response["choices"][0]["message"]
        if(response_message.get("function_call")):
            function_name = response_message["function_call"]["name"]
            args = json.loads(response_message.function_call.get('arguments'))

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
                if isEventStream:
                    event_data = {
                        "message": availableFunctionsDescription[function_name]
                    }
                    sse_data = f"data: {json.dumps(event_data)}\n\n"
                    yield sse_data
                result = function_to_call(**args)
                if(function_name=="generateSQLQuery"):
                    if isEventStream:
                        event_data = {
                            "message": "Querying database"
                        }
                        sse_data = f"data: {json.dumps(event_data)}\n\n"
                        yield sse_data
                    isSpeciesData, result = GetSQLResult(result)
                    break
                else:
                    modifiedMessages.append({"role":"function","content":result,"name": function_name})
                    #modifiedMessages.append({"role":"system","content":"Is the result generated in previous query enough to response the prompt. Prompt: {prompt} Output either 'True' or 'False', nothing else"})
                    #response2 = openai.ChatCompletion.create(
                    #    model="gpt-3.5-turbo-0613",
                    #    messages=modifiedMessages,
                    #    temperature=0,
                    #)
                    #if("TRUE" in response2["choices"][0]["message"]["content"].upper()):
                    #    modifiedMessages = modifiedMessages[:-2] 
                    #    break
                    #else:
                    #    modifiedMessages = modifiedMessages[:-1] 
            else:
                raise ValueError("No function named '{function_name}' in the global scope")
        else:
            break

    if isEventStream:
        event_data = {
            "message": "Formatting response"
        }
        sse_data = f"data: {json.dumps(event_data)}\n\n"
        yield sse_data

    summerizerResponse = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        messages=[{"role": "system","content":"""
        Based on the below details output a json in provided format. The response must be a json. The output json must be valid.
                   If the output is vegaLite, you must generate the schema
        
        {
            "outputType": "", //enum(image, text, table, heatmap, vegaLite, taxonomy) The data type based on the 'input' and previous response, use table when the data can be respresented as rows and column and when it can be listed out
            "summary": "", //Summary of the data based on the 'output', If there are no results, output will be None
            "vegaSchema": { // Visualization grammar, Optional, Only need when the input asks for visualization except heatmap
            }
        }

        """},{"role":"user", "content": "{\"input\": \"" + prompt + "\", \"output\":\"" + str(result)[:NUM_RESULTS_TO_SUMMARIZE] + "\"}"}],
    )
    try:
        print(summerizerResponse["choices"][0]["message"]["content"])
        summaryPromptResponse = json.loads(str(summerizerResponse["choices"][0]["message"]["content"]))
        output = {
            "outputType": summaryPromptResponse["outputType"],
            "responseText": summaryPromptResponse["summary"],
        }
        if("vegaSchema" in summaryPromptResponse):
            output["vegaSchema"] = summaryPromptResponse["vegaSchema"]

    except:
        print('summerizer failed')
        summaryPromptResponse = {}
        summaryPromptResponse["outputType"] = 'text'
        if isSpeciesData:
            summaryPromptResponse["outputType"] = 'image'
        if result!=None and len(result) > 0 and 'taxonomy' in result[0]:
            summaryPromptResponse["outputType"] = 'taxonomy'

        output = {
            "outputType": summaryPromptResponse["outputType"],
            "responseText": 'Here are the results',
            "vegaSchema": '',
        }

    if(isSpeciesData):
        #computedTaxonomicConcepts = []#adding taxonomy data to only the first species in the array with a given concept.
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
        
    if isEventStream:
        event_data = {
            "result": output
        }
        sse_data = f"data: {json.dumps(event_data)}\n\n"
        yield sse_data
    end_time = time.time()

    time_taken = end_time - start_time

    formatted_time = "{:.2f}".format(time_taken)
    print(f"Time taken: {formatted_time} seconds")

    return output



#DEBUG_LEVEL = 5

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
#print(json.dumps(get_Response("Find me images of creatures with tentacles in Monterey bay and depth less than 5k meters.")))
#print(json.dumps(get_Response("Find me images of moon jelliefish in Monterey bay and depth less than 5k meters")))
#print(json.dumps(get_Response("Find me images of rattails in Monterey bay and depth less than 5k meters")))

#for v in get_Response("Find me images of starfish in Monterey bay and depth less than 5k meters", isEventStream=True):
#for v in get_Response("Find me images of moon jellyfish in Monterey bay and depth less than 5k meters", isEventStream=True):
#for v in get_Response("Find me images of creatures with tentacles in Monterey bay and depth less than 5k meters", isEventStream=True):
#for v in get_Response("Find me images of ray-finned creatures in Monterey bay and depth less than 5k meters", isEventStream=True):
for v in get_Response("Find me images of Aurelia Aurita", isEventStream=True):
    print(v)

#test_msgs = [{'role': 'user', 'content': 'find me images of aurelia aurita'}, {'role': 'assistant', 'content': "{'outputType': 'image', 'responseText': 'Images of Aurelia Aurita', 'vegaSchema': {}, 'species': [{'url': 'https://fathomnet.org/static/m3/framegrabs/Ventana/images/3405/00_05_46_16.png', 'image_id': 2593314, 'concept': 'Aurelia aurita', 'id': 2593317}, {'url': 'https://fathomnet.org/static/m3/framegrabs/Ventana/images/3184/02_40_29_11.png', 'image_id': 2593518, 'concept': 'Aurelia aurita', 'id': 2593520}, {'url': 'https://fathomnet.org/static/m3/framegrabs/Doc%20Ricketts/images/0970/06_02_03_18.png', 'image_id': 2598130, 'concept': 'Aurelia aurita', 'id': 2598132}, {'url': 'https://fathomnet.org/static/m3/framegrabs/Ventana/images/3082/05_01_45_07.png', 'image_id': 2598562, 'concept': 'Aurelia aurita', 'id': 2598564}, {'url': 'https://fathomnet.org/static/m3/framegrabs/Doc%20Ricketts/images/0971/03_42_04_04.png', 'image_id': 2600144, 'concept': 'Aurelia aurita', 'id': 2600146}, {'url': 'https://fathomnet.org/static/m3/framegrabs/Ventana/images/3219/00_02_48_21.png', 'image_id': 2601105, 'concept': 'Aurelia aurita', 'id': 2601107}, {'url': 'https://fathomnet.org/static/m3/framegrabs/Ventana/images/3185/00_05_28_02.png', 'image_id': 2601178, 'concept': 'Aurelia aurita', 'id': 2601180}, {'url': 'https://fathomnet.org/static/m3/framegrabs/Ventana/images/3082/04_59_01_12.png', 'image_id': 2601466, 'concept': 'Aurelia aurita', 'id': 2601468}, {'url': 'https://fathomnet.org/static/m3/framegrabs/Ventana/images/3184/02_40_58_22.png', 'image_id': 2603507, 'concept': 'Aurelia aurita', 'id': 2603509}, {'url': 'https://fathomnet.org/static/m3/framegrabs/Ventana/stills/2000/236/02_33_01_18.png', 'image_id': 2604817, 'concept': 'Aurelia aurita', 'id': 2604819}]}"}]

#test_msgs = [{"prompt": 'Find me images of Moon jellyfish', "response": json.dumps({'a': '123', 'b': '456'})}, {"prompt": 'What do they look like', "response": json.dumps({'responseText': 'They are pink and translucent'})}]
#print(get_Response("Where can I find them", test_msgs))
#print(get_Response("What color are they", test_msgs))

#print(json.loads(getTaxonomyTree('Asteroidea')))

#print(json.dumps(get_Response('Find me the best images of Aurelia aurita')))
#print(json.dumps(get_Response('Find me images of creatures commonly found in the same images as Aurelia aurita in Monterey Bay')))
#print(json.dumps(get_Response('Find me images of Aurelia aurita that don’t have other creatures in them')))

#print(json.dumps(get_Response('Find me 3 images of moon jellyfish in Monterey bay and depth less than 5k meters')))

#print(json.dumps(get_Response('Find me images of Aurelia aurita sorted by depth')))
#print(json.dumps(get_Response('Find me images of creatures that are types of octopus in random order')))


#print(get_Response("Generate a heatmap of Aurelia aurita in Monterey Bay"))
