from decimal import Decimal
from .constants import *
from .utils import getScientificNames, isNameAvaliable, findDescendants, findAncestors, getParent, findRelatives, filterUnavailableDescendants, changeNumberToFetch, postprocess, fixTabsAndNewlines

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
import plotly
from server.models import Interaction
import plotly.graph_objects as go
import plotly.io as pio
import datetime
import struct

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
from torchvision import models, transforms
import base64
from PIL import Image
import io
import torch
from concurrent.futures import ThreadPoolExecutor

import os
os.environ["OPENAI_API_KEY"] = KEYS['openai']
openai.api_key = KEYS['openai']

model = models.efficientnet_b7(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def base64_to_pil_image(base64_string):
    image_data = base64.b64decode(base64_string)
    
    image_buffer = io.BytesIO(image_data)
    
    pil_image = Image.open(image_buffer)
    
    return pil_image

# ==== Taxonomy ====

def getTaxonomyTree(
    scientificName: str
) -> list:
    """Function to get the taxonomy tree contain the names and taxonomic ranks of the ancestors and descendants for a scientific name. 
    ONLY return a machine-readable JSON object, and nothing more."""
    
    taxonomy = {}
    with open("data/taxonomy.json") as f:
        taxonomy = json.load(f)
    if scientificName in taxonomy:
        taxonomy = taxonomy[scientificName]
        taxonomy['concept'] = scientificName
        #print(taxonomy)
        return json.dumps(taxonomy)
    
    descendants = filterUnavailableDescendants(findDescendants(scientificName, species_only=False))
    ancestors = findAncestors(scientificName)

    rank = ""
    for d in descendants:
        if d.name.lower() == scientificName.lower():
            rank = d.rank

    descendants = [{'name': d.name, 'rank': d.rank.lower(), 'parent': getParent(d.name)} for d in descendants if d.name.lower() != scientificName.lower()]
    ancestors = [{'name': d.name, 'rank': d.rank.lower()} for d in ancestors]
    ancestors.reverse()

    taxonomy = {'concept': scientificName, 'rank': rank.lower(), 'taxonomy': {'descendants': descendants, 'ancestors': ancestors}}
    print(taxonomy)

    return json.dumps(taxonomy)


def getTaxonomicRelatives(
    scientificName: str
) -> list:
    """Function to get the taxonomic relatives for a scientific name."""  
    relatives = findRelatives(scientificName)
    relatives = [d.name for d in relatives if d.name.lower() != scientificName.lower()]
    
    taxonomy = {'concept': scientificName, 'relatives': relatives}
    #print(taxonomy)
    
    return json.dumps(taxonomy)
    

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

def modifyExistingVisualization(prompt: str, plotlyCode: str, sampleData:str):

    code="#sample data: "+sampleData.replace("\n","")+"\n\n"+plotlyCode
    
    summerizerResponse = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            temperature=0,
            messages=[{"role": "system","content":"""
            Modify the python plotly code below based on the user's instruction. Donot output any comments. Re-generate the imports and the drawVisualization function.

            output in this JSON format
                       {
                        "plotlyCode":"",
                       "responseText": "",
                       }

            In the Plotly code, ensure all double quotation marks ("") are properly escaped with a backslash (). The input data object is just a list of object, if you want it to be pandas data frame object, convert it first. Donot use mapbox, use openstreet maps instead.
            The response text is a message to user saying that you made the modification. 
            Output only the json nothing else.
            """},{"role":"user", "content": "code: \n" + code + "\ninstruction:" + prompt+ "\nsample data:"+sampleData}],
        )
    
    result = summerizerResponse["choices"][0]["message"]["content"]
    result = result.strip()
    first_brace_position = result.find('{')
    if first_brace_position!=0:
        result = result[first_brace_position:]
    if result.endswith('```'):
        result = result[:-3]
    
    return json.loads(result)

    
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
        names = name.replace(', and ', ', ').replace(' and ', ', ').replace(' or ', ', ').split(', ')
        for n in names:
            if isNameAvaliable(n):
                results.append(n)
                continue
            results.extend(getScientificNames(n))

        if len(description) == 0:
            description = name
    
    if len(results) == 0:
        desc = list(set(name.split(' ') + description.split(' ')))
        if len(description) > 0:
            desc.append(description)
        desc = list(set([d for d in desc if len(d)>0]))
            
        #print(desc)
        for d in desc:
            results.extend(getScientificNames(d, False, SEMANTIC_MATCHES_JSON, True))
    
    if len(description) > 0 and len(results) == 0:
        results = list(getConceptCandidates(description))
    
    if len(results) > 0:
        if len(results) > LANGCHAIN_SEARCH_CONCEPTS_TOPN:
            results = results[:LANGCHAIN_SEARCH_CONCEPTS_TOPN]
        results = list(dict.fromkeys(results))
        print(results)
        if len(results)==1 and results[0]==name:
            print("Error: "+name+" is already a scientific name")
            return "Error: "+name+" is already a scientific name. Do not run this function again with the same input"
        return ", ".join(results)
    

    
    return "anything"


def getAnswer(
    question: str,
) -> list:
    """Function for questions about the features of a species.
    DO NOT use this tool for fetching images, taxonomy or generating charts"""

    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role":"system","content":"You are FathomGPT. You have access to fathom database that you can use to retrieve and visualize data of marine species. Answer the user's question using your own knowledge"},{"role":"user","content":question}],
            temperature=0,
        )
    return response["choices"][0]["message"]["content"]
    

# ==== SQL database query ====

def GetSQLResult(query: str, isVisualization: bool = False, imageData = None, prompt = "", fullGeneratedSQLJSON=None, isEventStream=False):
    isSpeciesData = False
    output = ""

    errorRunningSQL = False
    sqlGenerationTries = 3
    while(sqlGenerationTries>=0):
        if sqlGenerationTries==0:
            output="Error generating the sql for the prompt"
            errorRunningSQL = True
            break
        try:
            connection = pymssql.connect(
                server=KEYS['SQL_server'],
                user=KEYS['Db_user'],
                password=KEYS['Db_pwd'],
                database=KEYS['Db']
            )

            cursor = connection.cursor()

            if imageData != "" and imageData is not None:
                cursor.execute(addImageSearchQuery([imageData], fullGeneratedSQLJSON))
            else:
                cursor.execute(query)

            rows = cursor.fetchall()
            errorRunningSQL = False

            if isVisualization:
                results_dict = {header[0]: [] for header in cursor.description}

                if rows:
                    for row in rows:
                        for idx, value in enumerate(row):
                            column_header = cursor.description[idx][0]
                            if isinstance(value, bytes) and len(value) == 8:
                                try:
                                    # Assuming the binary format is specific to SQL Server's datetime
                                    timestamp = struct.unpack('Q', value)[0]
                                    value = datetime.datetime.fromtimestamp(timestamp / 1000)  # Convert to seconds
                                except struct.error:
                                    # Handle or log error if unpacking fails
                                    print("Could not unpack bytes to a timestamp.")
                            elif isinstance(value, bytes):
                                # Fallback to decode bytes to string for other byte objects
                                value = value.decode('utf-8', errors='ignore')
                            results_dict[column_header].append(value)
                else:
                    print("No rows found for this query.")
                    results_dict = None
                return (False, results_dict, False)

            if rows:
                columns = [column[0] for column in cursor.description]
                output = []
                for row in rows:
                    row_data = {}
                    for idx, value in enumerate(row):
                        if isinstance(value, bytes) and len(value) == 8:
                            try:
                                # Assuming the binary format is specific to SQL Server's datetime
                                timestamp = struct.unpack('Q', value)[0]
                                value = datetime.datetime.fromtimestamp(timestamp / 1000)  # Convert to seconds
                            except struct.error:
                                # Handle or log error if unpacking fails
                                print("Could not unpack bytes to a timestamp.")
                        elif isinstance(value, bytes):
                            # Fallback to decode bytes to string for other byte objects
                            value = value.decode('utf-8', errors='ignore')
                        elif isinstance(value, Decimal):
                            # Convert Decimal to float for numerical processing
                            value = float(value)
                        row_data[columns[idx]] = value
                    output.append(row_data)
                isSpeciesData = True
            else:
                print("No rows found for this query.")
                output = None
            break
        except pymssql.OperationalError as e:
            print("Database connection issues", e)
            if e.args[0]==229:
                output = "I don't have enough sufficient permission to execute your prompt."
            else:
                output = "Error connecting to the database"
            errorRunningSQL = True
            break
        except pymssql.ProgrammingError as e:
            sqlGenerationTries-=1
            if isEventStream:
                event_data = {
                    "message": "Error with the generated code. Fixing it. Try "+str(3-sqlGenerationTries)
                }
                sse_data = f"data: {json.dumps(event_data)}\n\n"
                yield sse_data
            if imageData == "":
                query = fixGeneratedSQL(prompt, query, str(e))
            else:
                fullGeneratedSQLJSON['sqlServerQuery'] = fixGeneratedSQLForImageSearch(fullGeneratedSQLJSON['sqlServerQuery'], query, str(e))
                print(fullGeneratedSQLJSON['sqlServerQuery'])
        except Exception as e:
            print("Error processing sql server response", e)
            output = "Error while retreiving or running the sql server query"
            errorRunningSQL = True
            break

    return (isSpeciesData, output, errorRunningSQL)

def tryFixGeneratedCode(prompt, code, error):
    completion = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            temperature=0,
            messages=[{"role": "system","content":"""
            An AI agent has generated a plotly code to show an output based on the user prompt. But the code has an error. Fix the plotly code error. The user provides the prompt, plotly code with error and the detaul of the error that has occurred. Generate only the plotly code nothing else donot output anything. The plotly code must have the necessary import and the drawVisualization function that takes in python list data and outputs plotly fig object.
            Output the full python code.
            In the Plotly code, ensure all double quotation marks ("") are properly escaped with a backslash (). The input data object is just a list of object, if you want it to be pandas data frame object, convert it first. Donot use mapbox, use openstreet maps instead.
            """},{"role":"user", "content": "user prompt: " + prompt+ "\n\nplotly code: \n" + code + "\n\nerror: "+error}],
        )
    
    result = completion.choices[0].message.content.strip()
    if result.startswith('```python'):
        result=result.replace('```python','')
    if result.endswith('```'):
        result = result[:-3]
    result = fixTabsAndNewlines(result)
    return result


def generatesqlServerQuery(
    prompt: str,
    scientificNames: str,
    name: str,
    outputType: str,
    inputImageDataAvailable: bool
) -> str:
    """Converts text to sql. If the common name of a species is provided, it is important convert it to its scientific name. If the data is need for specific task, input the task too. The database has image, bounding box and marine regions table. The database has data of species in a marine region with the corresponding images.
    """

    print("prompt sent to generatesqlServerQuery ",prompt)

    if inputImageDataAvailable and not isNameAvaliable(name):
        prompt = prompt.replace(name, '')
    else:
        if isinstance(name, str) and len(name) > 1:
            prompt = prompt.replace(name, scientificNames)
        elif isinstance(scientificNames, str) and len(scientificNames) > 1:
            prompt = prompt.rstrip(".")+" with names: "+scientificNames

    needsGpt4 = outputType == "visualization"

    messages = [{"role": "system", "content": FEW_SHOT_DATA["imagesWithInput"]['instructions'] if inputImageDataAvailable else FEW_SHOT_DATA["visualization"]['instructions'] if needsGpt4 else """You are a very intelligent json generated that can generate highly efficient sql queries and plotly python code. You will be given an input prompt for which you need to generated the JSON in a format given below, nothing else.
                The Generated SQL must be valid
                The JSON format and the attributes on the JSON are provided below
                {
                    "sqlServerQuery": "",
                 """+ ("""
                    "sampleData": "",
                    """ if needsGpt4 else "")
                    +
                """
                    "responseText": ""
                }

                sqlServerQuery: This is the sql server query you need to generate based on the user's prompt. The database structure provided will be very useful to generate the sql query. sqlServerQuery must be generated when InputImageDataAvailable is True 
                """+
                ("""
                sampleData: This is the sample data that you think should be generated when running the sql query. This is optional. It is only needed when the outputType is visualization
                """ if needsGpt4 else "")
                +"""responseText: Suppose you are answering the user with the output from the prompt. You need to write the message in this section. When the response is text, you need to output the textResponse in a way the values from the generated sql can be formatted in the text


                SQL Server Database Structure:
             """+DB_STRUCTURE+"\n\n"+
                FEW_SHOT_DATA[outputType]['instructions']
            }]
    messages.append({
            "role": "user","content": FEW_SHOT_DATA["imagesWithInput" if inputImageDataAvailable else outputType]['user']
        })
    messages.append({
            "role": "assistant","content": FEW_SHOT_DATA["imagesWithInput" if inputImageDataAvailable else outputType]['assistant']
        })
    messages.append({
            "role": "user","content": FEW_SHOT_DATA["imagesWithInput" if inputImageDataAvailable else outputType]['user2']
        })
    messages.append({
            "role": "assistant","content": FEW_SHOT_DATA["imagesWithInput" if inputImageDataAvailable else outputType]['assistant2']
        })

    messages.append({
                "role": "user","content": f"""
                User Prompt: {prompt}"""
            })
    print('gpt-3.5-turbo-0125' if needsGpt4 else SQL_IMAGE_SEARCH_FINE_TUNED_MODEL if inputImageDataAvailable else SQL_FINE_TUNED_MODEL)

    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0125' if needsGpt4 else SQL_IMAGE_SEARCH_FINE_TUNED_MODEL if inputImageDataAvailable else SQL_FINE_TUNED_MODEL,
        messages=messages,
        temperature=0
    )


    result = completion.choices[0].message.content
    first_brace_position = result.find('{')
    if first_brace_position!=0:
        result = result[first_brace_position:]
    if result.endswith('```'):
        result = result[:-3]
    result = fixTabsAndNewlines(result)
    result = json.loads(result)
    print(result)
    result['outputType'] = outputType
    return result

def addImageSearchQuery(imageData, generatedJSON):
    imageIDs = generatedJSON['similarImageIDs'] if 'similarImageIDs' in generatedJSON else []
    boundingBoxIDs = generatedJSON['similarBoundingBoxIDs'] if 'similarBoundingBoxIDs' in generatedJSON else []

    sql = ""

    if len(imageData)!=0:
        sql += """
            IF OBJECT_ID('tempdb..#InputFeatureVectors') IS NOT NULL DROP TABLE #InputFeatureVectors;
            IF OBJECT_ID('tempdb..#InputMagnitudes') IS NOT NULL DROP TABLE #InputMagnitudes;

            CREATE TABLE #InputFeatureVectors (
                image_id INT,
                vector_index INT,
                vector_value DECIMAL(18,5)
            );
        """
        for singleImageData in imageData:
            sql += "INSERT INTO #InputFeatureVectors (image_id, vector_index, vector_value) VALUES "+singleImageData+";\n"

        sql+="""
            CREATE TABLE #InputMagnitudes (
                image_id INT,
                magnitude DECIMAL(18,5)
            );

            INSERT INTO #InputMagnitudes (image_id, magnitude)
            SELECT 
                image_id,
                SQRT(SUM(POWER(vector_value, 2))) AS magnitude
            FROM #InputFeatureVectors
            GROUP BY image_id;

            WITH ImageSimialritySearch AS (
                SELECT
                    IFV.image_id AS bb1,
                    BBFV.bounding_box_id AS bb2,
                    SUM(IFV.vector_value * BBFV.vector_value) / (IM.magnitude * BB.magnitude) AS CosineSimilarity
                FROM #InputFeatureVectors IFV
                INNER JOIN bounding_box_image_feature_vectors BBFV ON IFV.vector_index = BBFV.vector_index
                INNER JOIN bounding_boxes BB ON BBFV.bounding_box_id = BB.id
                INNER JOIN #InputMagnitudes IM ON IFV.image_id = IM.image_id
                WHERE IM.magnitude > 0 AND BB.magnitude > 0
                GROUP BY IFV.image_id, BBFV.bounding_box_id, IM.magnitude, BB.magnitude
            )
            """
        
    if(len(imageIDs)!=0 or len(boundingBoxIDs)!=0):
        sql+="""

            DECLARE @BoundingBoxIDs TABLE (ID INT);
        """
        if(len(boundingBoxIDs)!=0):
            sql+="\nINSERT INTO @BoundingBoxIDs VALUES "+", ".join(f"({num})" for num in boundingBoxIDs)+";"
        if(len(imageIDs)!=0):
            sql+="\nDECLARE @ImageIDs TABLE (ID INT);"
            sql+="\nINSERT INTO @ImageIDs VALUES "+", ".join(f"({num})" for num in imageIDs)+";"
            sql+="""
                DECLARE @BoundingBoxIDs TABLE (ID INT);

                INSERT INTO @BoundingBoxIDs (ID)
                SELECT BB.id
                FROM bounding_boxes BB
                WHERE EXISTS (
                    SELECT 1
                    FROM @ImageIDs I
                    WHERE BB.image_id = I.ID
                );

                WITH InputFeatureVectors AS (
                    SELECT
                        BBI.ID AS InputBoxID,
                        BBFV.vector_index,
                        BBFV.vector_value
                    FROM @BoundingBoxIDs BBI
                    INNER JOIN bounding_box_image_feature_vectors BBFV ON BBI.ID = BBFV.bounding_box_id
                ),
                SimilaritySearch AS (
                    SELECT
                        IFV.InputBoxID AS bb1,
                        BBFV.bounding_box_id AS bb2,
                        SUM(IFV.vector_value * BBFV.vector_value) / (IM.magnitude * TM.magnitude) AS CosineSimilarity
                    FROM InputFeatureVectors IFV
                    INNER JOIN bounding_box_image_feature_vectors BBFV ON IFV.vector_index = BBFV.vector_index
                    INNER JOIN bounding_boxes IM ON IFV.InputBoxID = IM.id
                    INNER JOIN bounding_boxes TM ON BBFV.bounding_box_id = TM.id
                    WHERE IFV.InputBoxID != BBFV.bounding_box_id
                    AND IM.magnitude > 0 AND TM.magnitude > 0
                    GROUP BY IFV.InputBoxID, BBFV.bounding_box_id, IM.magnitude, TM.magnitude
                )

            """
    if len(imageData)!=0 and (len(imageIDs)!=0 or len(boundingBoxIDs)!=0):
        sql+="SELECT TOP 10 FROM (\n"
    if len(imageData)!=0:
        genSQLQuery = generatedJSON['sqlServerQuery'].replace('SimilaritySearch', 'ImageSimialritySearch')
        sql+=genSQLQuery
    if len(imageData)!=0 and (len(imageIDs)!=0 or len(boundingBoxIDs)!=0):
        sql+="\nUNION ALL\n"
    if len(imageIDs)!=0 or len(boundingBoxIDs)!=0:
        sql+=genSQLQuery
    if len(imageData)!=0 and (len(imageIDs)!=0 or len(boundingBoxIDs)!=0):
        sql+=") ORDER BY CosineSimilarity DESC"
    return sql



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
        genTool(generatesqlServerQuery),
        #genTool(GetSQLResult),
        genTool(getTaxonomyTree),
        genTool(getTaxonomicRelatives),
        genTool(getAnswer),
        #genTool(getOtherCreaturesInImage),
        #genTool(getImageQualityScore),
        
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

def fixGeneratedSQL(prompt, sql, error):
    response = openai.ChatCompletion.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system","content":f"""
                You are a text to sql generator. A sql query has been generated for a given prompt. Unforunately, user got an error while running the sql query. Your task is to fix the error in the sql query. User will provide the prompt, the sql query that has error and the error detail.
                You need to generate only the sql query nothing else. Donot generate any comments in the sql query. Only sql query needs to be generated.
             
                The SQL Server is a Microsoft SQL Server with the following database structure:
                {DB_STRUCTURE}

                Only output the fixed sql query nothing else.
             """},
            {"role":"user", "content": "user prompt: " + prompt+ "\n\nsql server query: " + sql + "\n\nerror: "+error}
        ],
        functions=availableFunctions,
        function_call="auto",
        temperature=0,
    )

    result = response.choices[0].message.content.strip()
    if result.startswith('```sql'):
        result=result.replace('```sql','')
    if result.endswith('```'):
        result = result[:-3]

    return result

def fixGeneratedSQLForImageSearch(prompt, sql, error):
    response = openai.ChatCompletion.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system","content":f"""
                You are a very intelligent text to sql generator.
                The prompt will asks for similar images, there is another system that takes in the similarImageIDs and similarBoundingBoxIDs and does the similarity search. Thus, now you have sql table SimilaritySearch that has the input bounding box id as bb1, output bounding box id as bb2 and Cosine Similarity Score as CosineSimilarity. You will use this table and add the conditions that is given provided by the user. You will also ouput the ouput bounding box image url and the concept. The result must be ordered in descending order using the CosineSimilarity value. Also, you will take 10 top results unless specified by the prompt
                
                The SQL Server is a Microsoft SQL Server with the following database structure:
                {DB_STRUCTURE}
             
                Only output the fixed sql query nothing else.
             """},
            {"role":"user", "content": "user prompt: " + prompt+ "\n\nsql server query: " + sql + "\n\nerror: "+error}
        ],
        functions=availableFunctions,
        function_call="auto",
        temperature=0,
    )

    result = response.choices[0].message.content.strip()
    if result.startswith('```sql'):
        result=result.replace('```sql','')
    if result.endswith('```'):
        result = result[:-3]

    return result

availableFunctions = [{
    "name": "getScientificNamesFromDescription",
    "description": "Function to get all scientific names that fits a common name or appearance. If there are no matches, return anything. DO NOT use this tool for descriptions of location, depth, taxonomy, salinity, or temperature. If the input name is already a scientific name, the function will return the same name, donot run the same function with the same input more than once.",
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
    "name": "generatesqlServerQuery",
    "description": "Converts text to sql. If no scientific name of a species is provided, it is important convert it to its scientific name. If the data is need for specific task, input the task too. The database has image, bounding box and marine regions table. The database has data of species in a marine region with the corresponding images. The sql server has can search images of species.",
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
            "outputType": {
                "type": "string",
                "description": """enum of 'images', 'text', 'visualization','table'. The enum type must be deduced based on the prompt. 
                If the prompt says to generate or find image, not the image count, outputType = images, 
                else if prompt says to generate any kind of visualization like graph, plots,etc., outputType = visualization, 
                else if the prompt asks about two or more type of species data or the output data might contain data of two or more type of species, or asks about image count or something numerical of more than one species, outputType = table, 
                else outputType = text"""
            },
        },
        "required": ["prompt", "scientificNames", "name", "outputType"],
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
},{
    "name": "getAnswer",
    "description": "This tool has general knowledge about the world. Use this tool for questions about a species. DO NOT use this tool for fetching images, taxonomy or generating charts.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A question with a short and simple answer",
            },
        },
        "required": ["question"],
    },
},
{
    "name": "modifyExistingVisualization",
    "description": "Important: Donot run this function when the prompt says to generate a visualization. This tool modifies the visualization. This tool only makes visual changes to the chart. If the data itself needs to be changed, do not use this tool.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Prompt that instructs what modification that needs to be done in the chart",
            },
        },
        "required": ["prompt"],
    },
}
]

availableFunctionsDescription = {
    "getScientificNamesFromDescription": "Generating scientific name from description",
    
    "generatesqlServerQuery": "Generating SQL Query",
    
    "getTaxonomyTree": "Getting the taxonomy tree",
    
    "getTaxonomicRelatives": "Getting taxonomic relatives",
    
    "getAnswer": "Getting answer from ChatGPT.",

    "modifyExistingVisualization": "Modifying the chart.",

}

# messages must be in the format: [{"prompt": prompt, "response": json.dumps(response)}]
def get_Response(prompt, imageData="", messages=[], isEventStream=False, db_obj=None):


    initialMessagesCount = len(messages)
    start_time = time.time()
    messages.insert(0,{"role":"system", "content":"""You are FathomGPT. You have access to fathom database that you can use to retrieve and visualize data of marine species. 
                       You have the ability to do visulizations like generating area chart showing the year the images were taken, generating heatmap of species in Monterey Bay, etc.
                       You have the ability to find marine species that live below 1000 meters, find the total count of marine species in Monterey Bay, etc.

                       Use the tools provided to generate response to the prompt. Important: If the prompt contains a common name or description use the 'getScientificNamesFromDescription' tool first. If the prompt is for similar images, use the 'getScientificNamesFromDescription' tool last. The 'getScientificNamesFromDescription' function will output the same input name when the input name is already a scientific name. Donot re-run the function with the same input. The prompt might have refernce to previous prompts but the tools do not have previous memory. So do not use words like their, its in the input to the tools, provide the name. 
                       If you are not running any function, just output the text, dont format it like the other content
                       Work on the task until you finish, do not ask if you should continue
                       """})

    messages.append({"role":"user","content":("User has provided image of species most probably to do an image search" if imageData!="" else "")+"\nPrompt:"+prompt})
    isSpeciesData = False
    result = None
    function_name=""
    curLoopCount = 0
    allResultsCsv = []
    allResults = {}

    if SAVE_INTERMEDIATE_RESULTS:
        with open('data/intermediate_results.json') as f:
            allResultsCsv = json.load(f)

    if isEventStream:
        event_data = {
            "message": "Evaluating Prompt"
        }
        sse_data = f"data: {json.dumps(event_data)}\n\n"
        yield sse_data

    eval_image_feature_string = ""
    if imageData is not None and imageData!="":
        pil_image=base64_to_pil_image(imageData)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        tensor = preprocess(pil_image)
        with torch.no_grad():
            features = model(tensor.unsqueeze(0))
        
        features_squeezed = features.squeeze()
        formatted_features = []

        for index, feature in enumerate(features_squeezed, start=1):
            formatted_feature = f"(-1, {index}, {feature:.5f})"
            formatted_features.append(formatted_feature)

        eval_image_feature_string = ", ".join(formatted_features)
    
    
    while curLoopCount < 4:
        curLoopCount+=1
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            functions=availableFunctions,
            function_call="auto",
            temperature=0,

        )
        response_message = response["choices"][0]["message"]
        if(response_message.get("function_call")):
            function_name = response_message["function_call"]["name"]
            args = json.loads(response_message.function_call.get('arguments'))
            
            print('----')
            print(function_name)

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
            if(function_name=="modifyExistingVisualization"):
                if isEventStream:
                    event_data = {
                        "message": "Modifying the visualization"
                    }
                    sse_data = f"data: {json.dumps(event_data)}\n\n"
                    yield sse_data

                lastInteractionWithVisualization = None
                for i in range(initialMessagesCount, 0, -1):
                    try:
                        lastPlotlyCode = eval(messages[i]['content'])['plotlyCode']
                        if lastPlotlyCode is not None and lastPlotlyCode != "":
                            lastInteractionWithVisualization = json.loads(json.dumps(messages[i]))
                            break
                    except Exception:
                        continue
                
                if lastInteractionWithVisualization is None:
                    messages.append({"role":"function","content":"User Prompt Error. No previous visualization data found","name": function_name})
                    continue
                parsedPrvresponse = eval(lastInteractionWithVisualization['content'])

                prvPlotlyCode = parsedPrvresponse['plotlyCode']
                sampleData = parsedPrvresponse['sampleData']

                evalNewdbQueryNeeded = openai.ChatCompletion.create(
                    model="gpt-4-turbo-preview",
                    temperature=0,
                    messages=[{"role": "system","content":"""
                    Your task is to evaluate the current situation and come to a conclusion of True or False. You will be given a plotly code. 
                               The plotly code will have a sample data as a comment. This is the current data the system has right now. User is trying to modify the plotly visualization. What user is trying to acheieve is given by the prompt that the user provides.
                               Your task is to evaluate either the user needs additional data or not to perform the modification according to the prompt to the visualization.
                               Plotly code will be modified later by the user based on the user's prompt, if there are any visual changes like change the x, y axis range, visual attributes, another type of visualization, etc that will be done by the user. In those cases the sample data does not need to be modified.

                    Output True if user needs additional data to modify the visualization else Output False
                    """},{"role":"user", "content": "code: \n" + "#sample data: "+sampleData.replace("\n","")+"\n\n"+prvPlotlyCode + "prompt:" + args['prompt']}],
                )

                if("TRUE" in evalNewdbQueryNeeded["choices"][0]["message"]["content"].upper()):
                    messages.append({"role":"function","content":"Not enough data to modify the visualization, using previous prompts, re-generate another prompt that instructs to draw the visualization with the modification and run generatesqlServerQuery function. Make sure to add all the constraints that were in the earlier prompts.","name": function_name})
                    continue

                def first_task():
                    return function_to_call(prompt=args['prompt'], plotlyCode=prvPlotlyCode, sampleData=sampleData)

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future_result = executor.submit(first_task)
                    
                    isSpeciesData, sqlResult, errorRunningSQL = yield from GetSQLResult(parsedPrvresponse['sqlServerQuery'], True, prompt=prompt, isEventStream=isEventStream)

                    result = future_result.result()

                if errorRunningSQL:
                    event_data = {
                            "result": {
                                "outputType": "error",
                                "responseText": sqlResult
                            }
                        }
                    sse_data = f"data: {json.dumps(event_data)}\n\n"
                    yield sse_data
                    Interaction.objects.create(main_object=db_obj, request=prompt, response="Error while running the generated sql query")
                    return None
                
                codeGenerationTries = 3
                fig=None
                while(codeGenerationTries > 0):
                    try:
                        exec(result["plotlyCode"], globals())
                        fig = drawVisualization(sqlResult)
                        codeGenerationTries = 0
                    except Exception as e:
                        codeGenerationTries-=1
                        if(codeGenerationTries==0):
                            event_data = {
                                    "result": {
                                        "outputType": "error",
                                        "responseText": "Error running the generated code"
                                    }
                                }
                            sse_data = f"data: {json.dumps(event_data)}\n\n"
                            yield sse_data
                            Interaction.objects.create(main_object=db_obj, request=prompt, response="Error running the generated plotly code")
                            return None
                        event_data = {
                                "message": "Error with the generated code. Fixing it. Try "+str(3-codeGenerationTries)
                            }
                        sse_data = f"data: {json.dumps(event_data)}\n\n"
                        yield sse_data
                        result["plotlyCode"] = tryFixGeneratedCode(prompt, result["plotlyCode"], str(e))

                html_output = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
                output = {}
                output["outputType"] = parsedPrvresponse['outputType']
                output["responseText"] = result["responseText"]
                output["html"] = html_output

                if isEventStream:
                    output['guid']=str(db_obj.id)
                    event_data = {
                        "result": output
                    }
                    sse_data = f"data: {json.dumps(event_data)}\n\n"
                    yield sse_data
                    output["sampleData"] = parsedPrvresponse['sampleData']
                    output["plotlyCode"] = prvPlotlyCode
                    output["sqlServerQuery"] = parsedPrvresponse['sqlServerQuery']
                    output["html"] = ""
                    Interaction.objects.create(main_object=db_obj, request=prompt, response=output)
                return
            elif function_to_call:
                if isEventStream:
                    event_data = {
                        "message": availableFunctionsDescription[function_name],
                    }
                    if isinstance(result, str):
                        event_data["message"] = event_data["message"]
                        allResults[function_name] = result
                    print(event_data)
                    sse_data = f"data: {json.dumps(event_data)}\n\n"
                    yield sse_data
                if(function_name=="generatesqlServerQuery"):
                    args["inputImageDataAvailable"] = len(eval_image_feature_string)!=0
                
                result = function_to_call(**args)
                


                if(function_name=="generatesqlServerQuery"):
                    if 'sqlQuery' in result:
                        result['sqlServerQuery'] = result['sqlQuery']

                    if "sqlServerQuery" not in result or result['sqlServerQuery'] == "":
                        continue

                    def gen_plotly_task(vizprompt, sampleData):
                        vizmessages = [{"role": "system","content":"""
                            Your task is to generate plotly code based on the user's prompt. The plotly code should define the necessary import and have drawVisualization function defined that takes in data variable and outputs plotly visualization object.
                            The input data object is just a list of object, if you want it to be pandas data frame object, convert it first. Donot use mapbox, use openstreet maps instead.
                            The plotly visualization should be able to take the data defined like the sample data and should not bug out for input same as sample data.
                            Important: The plotly code will be run with exact input as that of the sample data, so do not use any properties other than those defined in the sample data.
                            Do not write any comments in the code.
                            """}]

                        vizmessages.append({"role":"user","content": "prompt:" + FEW_SHOT_DATA['visualization']['user']+ "\nsample data:"+FEW_SHOT_DATA['visualization']['sampleData']})
                        vizmessages.append({"role":"user","content": "prompt:" + FEW_SHOT_DATA['visualization']['plotlyCode']})
                        vizmessages.append({"role":"user","content": "prompt:" + FEW_SHOT_DATA['visualization']['user2']+ "\nsample data:"+FEW_SHOT_DATA['visualization']['sampleData2']})
                        vizmessages.append({"role":"user","content": "prompt:" + FEW_SHOT_DATA['visualization']['plotlyCode2']})
                        vizmessages.append({"role":"user","content": "prompt:" + FEW_SHOT_DATA['visualization']['user3']+ "\nsample data:"+FEW_SHOT_DATA['visualization']['sampleData3']})
                        vizmessages.append({"role":"user","content": "prompt:" + FEW_SHOT_DATA['visualization']['plotlyCode3']})
                        vizmessages.append({"role":"user","content": "prompt:" + vizprompt+ "\nsample data:"+sampleData})

                        plotlyCodeGenerator = openai.ChatCompletion.create(
                            model="gpt-4-0125-preview",
                            temperature=0,
                            messages=vizmessages,
                        )

                        plotlyCode = plotlyCodeGenerator["choices"][0]["message"]["content"]
                        plotlyCode = plotlyCode.strip()
                        if plotlyCode.startswith('```python'):
                            plotlyCode=plotlyCode.replace('```python','')
                        if plotlyCode.endswith('```'):
                            plotlyCode = plotlyCode[:-3]
                        return plotlyCode

                    sql = result['sqlServerQuery']
                    limit = -1
                    if sql.strip().startswith('SELECT '):
                        limit, sql = changeNumberToFetch(sql)
                    if isEventStream:
                        event_data = {
                            "message": "Querying database...                             SQL Query:"+sql
                        }
                        sse_data = f"data: {json.dumps(event_data)}\n\n"
                        yield sse_data
                    if(result["outputType"]=="visualization"):
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future_result = executor.submit(gen_plotly_task, args['prompt'], result['sampleData'])
                            isSpeciesData, sqlResult, errorRunningSQL = yield from GetSQLResult(sql, result["outputType"]=="visualization", imageData=eval_image_feature_string, prompt=prompt, fullGeneratedSQLJSON=result,isEventStream=isEventStream)
                        
                            result["plotlyCode"] = future_result.result()
                            print(result["plotlyCode"])

                    else:
                        isSpeciesData, sqlResult, errorRunningSQL = yield from GetSQLResult(sql, result["outputType"]=="visualization", imageData=eval_image_feature_string, prompt=prompt, fullGeneratedSQLJSON=result,isEventStream=isEventStream)


                    if errorRunningSQL:
                        event_data = {
                                "result": {
                                    "outputType": "error",
                                    "responseText": sqlResult
                                }
                            }
                        sse_data = f"data: {json.dumps(event_data)}\n\n"
                        yield sse_data
                        Interaction.objects.create(main_object=db_obj, request=prompt, response="Error running the generated sql query")
                        return None
                    
                    if sqlResult is None or len(sqlResult)==0:
                        messages.append({"role":"function","content":"No results found after running the sql query. If getScientificNamesFromDescription was run earlier, ask user to specify the name of the species or any other description.","name": function_name})
                        continue



                        
                    print("got data from db")

                    try:
                        sqlResult, isSpeciesData = postprocess(sqlResult, limit, prompt, sql, isSpeciesData, args["scientificNames"], args["inputImageDataAvailable"])
                    except:
                        print('postprocessing error')
                        pass


                    if sqlResult and limit != -1 and limit < len(sqlResult) and isinstance(sqlResult, list):
                        sqlResult  = sqlResult[:limit]

                    if(result["outputType"]=="visualization"):
                        codeGenerationTries = 3
                        fig=None
                        while(codeGenerationTries > 0):
                            try:
                                exec(result["plotlyCode"], globals())
                                fig = drawVisualization(sqlResult)
                                codeGenerationTries = 0
                            except Exception as e:
                                print("Error ",str(e))
                                codeGenerationTries-=1
                                if(codeGenerationTries==0):
                                    event_data = {
                                            "result": {
                                                "outputType": "error",
                                                "responseText": "Error running the generated code"
                                            }
                                        }
                                    sse_data = f"data: {json.dumps(event_data)}\n\n"
                                    yield sse_data
                                    Interaction.objects.create(main_object=db_obj, request=prompt, response="Error running the generated plotly code")
                                    return None
                                event_data = {
                                        "message": "Error with the generated code. Fixing it. Try "+str(3-codeGenerationTries)
                                    }
                                sse_data = f"data: {json.dumps(event_data)}\n\n"
                                yield sse_data
                                result["plotlyCode"] = tryFixGeneratedCode(prompt, result["plotlyCode"], str(e))

                        #html_output = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                        html_output = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
                        result["html"]=html_output

                    elif(result["outputType"]=="table"):
                        result["table"]=sqlResult

                    elif(result["outputType"]=="images"):
                        print("species, ", sqlResult)
                        result["species"]=sqlResult

                    try:
                        result["responseText"]= result["responseText"].format(**sqlResult[0])
                    except:
                        print("Warining: Issues formatting data to response text")
                    break

                else:
                    messages.append({"role":"function","content":result,"name": function_name})
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
            if(function_name!="getTaxonomyTree" and function_name!="getTaxonomicRelatives"):
                try:
                    responseContent = response["choices"][0]["message"]['content']
                    print(responseContent)
                    if(eval(responseContent)["outputType"]=="text"):
                        result = eval(responseContent)
                    else:
                        messages.append({"role":"user","content":"You need to run the functions available to generate response to user's query."})
                        continue
                except:
                    try:
                        result = {
                            "outputType": "text",
                            "responseText": response["choices"][0]["message"]['content']
                        }
                    except:
                        result = {
                            "outputType": "text",
                            "responseText": response
                        }
            break

    output = None
    if(function_name=="getTaxonomyTree" or function_name=="getTaxonomicRelatives"):
        parsedResult = json.loads(result)

        taxonomyResponse = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role":"system","content":"Generate a short summary for this response for the prompt provided"},{"role":"user","content":"Prompt:"+prompt+"\nResponse+"+result}],
            temperature=0,
        )
        output = {
            "outputType": "taxonomy" if function_name=="getTaxonomyTree" else "text",
            "responseText": taxonomyResponse["choices"][0]["message"]["content"],
            "species": parsedResult if isinstance(parsedResult, list) else [parsedResult]
        }
    else:
        output = {
            "outputType": result["outputType"] if "outputType" in result else "",
            "responseText": result["responseText"] if "responseText" in result else "",
            "species": result["species"] if "species" in result else "",
            "html": result["html"] if "html" in result else "",
            "table": result["table"] if "table" in result else "",
        }



    #if isEventStream:
    #    event_data = {
    #        "message": "Formatting response"
    #    }
    #    sse_data = f"data: {json.dumps(event_data)}\n\n"
    #    yield sse_data
#
    #summerizerResponse = openai.ChatCompletion.create(
    #    model="gpt-3.5-turbo-0613",
    #    temperature=0,
    #    messages=[{"role": "system","content":"""
    #    Based on the below details output a json in provided format. The response must be a json. The output json must be valid.
    #            If the output is vegaLite, you must generate the schema
    #    
    #    {
    #        "outputType": "", //enum(image, text, table, heatmap, vegaLite, taxonomy) The data type based on the 'input' and previous response, must use "heatmap" as outputType when input says heatmap, use table when the data can be respresented as rows and column and when it can be listed out
    #        "summary": "", //Summary of the data based on the 'output', If there are no results, output will be None
    #        "vegaSchema": { // Visualization grammar, Optional, Only need when the input asks for visualization except heatmap
    #        }
    #    }
#
    #    """},{"role":"user", "content": "{\"input\": \"" + prompt + "\", \"output\":\"" + str(result)[:NUM_RESULTS_TO_SUMMARIZE] + "\"}"}],
    #)
#
    #    
    #try:
    #    summaryPromptResponse = json.loads(str(summerizerResponse["choices"][0]["message"]["content"]))
    #    output = {
    #        "outputType": summaryPromptResponse["outputType"],
    #        "responseText": summaryPromptResponse["summary"],
    #    }
    #    if(summaryPromptResponse["outputType"] == "vegaLite"):
    #        output["vegaSchema"] = summaryPromptResponse["vegaSchema"]
    #        output["vegaSchema"]["data"]["values"] = result
#
    #except:
    #    print('summerizer failed')
    #    summaryPromptResponse = {}
    #    summaryPromptResponse["outputType"] = 'text'
    #    if isSpeciesData:
    #        summaryPromptResponse["outputType"] = 'image'
    #    if result!=None and len(result) > 0 and 'taxonomy' in result[0]:
    #        summaryPromptResponse["outputType"] = 'taxonomy'
#
    #    output = {
    #        "outputType": summaryPromptResponse["outputType"],
    #        "responseText": 'Here are the results',
    #        "vegaSchema": '',
    #    }
    #if "heatmap" in prompt:
    #    summaryPromptResponse["outputType"] = "heatmap"
    #
    #if "getScientificNamesFromDescription" in allResults and "getAnswer" not in allResults:
    #    output["responseText"] = output["responseText"] + "\nScentific names: " + allResults["getScientificNamesFromDescription"]
#
    #if summaryPromptResponse["outputType"] == 'image' and result == None:
    #    output["outputType"] = 'text' 
    #    output["responseText"] = 'No data found in the database' 
    #if(isSpeciesData):
    #    #computedTaxonomicConcepts = []#adding taxonomy data to only the first species in the array with a given concept.
    #    #if isinstance(result, dict) or isinstance(result, list):
    #    #    for specimen in result:
    #    #        if "concept" in specimen and isinstance(specimen["concept"], str) and len(specimen["concept"]) > 0 and specimen["concept"] not in computedTaxonomicConcepts:
    #    #            taxonomyResponse = json.loads(getTaxonomyTree(specimen["concept"]))
    #    #            specimen["rank"] = taxonomyResponse["rank"]
    #    #            specimen["taxonomy"] = taxonomyResponse["taxonomy"]
    #    #            computedTaxonomicConcepts.append(specimen["concept"])
    #    output["species"] = result
#
    #elif(summaryPromptResponse["outputType"]=="taxonomy"):
    #    if(isinstance(result, list)):
    #        output["species"] = result
    #    else:
    #        output["species"] = [result]
    #    output["outputType"] = "species"
    #elif(summaryPromptResponse["outputType"]!="vegaSchema"):
    #    output["table"] = result
        
    if isEventStream:
        output['guid']=str(db_obj.id)
        event_data = {
            "result": output
        }
        sse_data = f"data: {json.dumps(event_data)}\n\n"
        yield sse_data
        output['html']=""
        if "species" in output:
            output['result']=output['species']
        output['species']=""
        output['sqlServerQuery']=result["sqlServerQuery"] if "sqlServerQuery" in result else ""
        output['plotlyCode']=result["plotlyCode"] if "plotlyCode" in result else ""
        output['sampleData']=result["sampleData"] if "sampleData" in result else ""
        Interaction.objects.create(main_object=db_obj, request=prompt, response=output)

    end_time = time.time()

    time_taken = end_time - start_time

    formatted_time = "{:.2f}".format(time_taken)
    print(f"Time taken: {formatted_time} seconds")
    
    if SAVE_INTERMEDIATE_RESULTS:
        with open("data/intermediate_results.json", "w") as outfile:
            row = {'prompt': prompt}
            i = 0
            for f in allResults:
                i = i + 1
                row['function'+str(i)] = f
                row['result'+str(i)] = allResults[f]
            allResultsCsv.append(row)
            json.dump(allResultsCsv, outfile)

    return output



#DEBUG_LEVEL = 5
SAVE_INTERMEDIATE_RESULTS = False

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
#for v in get_Response("Find me images of moon jelly or Pycnopodia helianthoides", isEventStream=True):
#for v in get_Response("What color is Aurelia aurita", isEventStream=True):
#for v in get_Response("What is the total number of images of Aurelia aurita in Monterey bay in the database", isEventStream=True):
#for v in get_Response("What are the ancestors of moon jelly", isEventStream=True):
#for v in get_Response("What species belong to the genus Aurelia", isEventStream=True):
#for v in get_Response("Show me the taxonomy of moon jelly", isEventStream=True):
#for v in get_Response("Display a bar chart showing the temperature ranges for Aurelia Aurita and Pycnopodia helianthoides from 0°C to 20 in 5°C increments", isEventStream=True):
#for v in get_Response("Find me similar images of jellyfish", isEventStream=True, imageData="/9j/4AAQSkZJRgABAQEASABIAAD/4QCQRXhpZgAASUkqAAgAAAAGABIBAwABAAAAAQAAABoBBQABAAAAVgAAABsBBQABAAAAXgAAACgBAwABAAAAAgAAADEBAgANAAAAZgAAADIBAgAUAAAAdAAAAAAAAABIAAAAAQAAAEgAAAABAAAAR0lNUCAyLjEwLjM0AAAyMDI0OjAyOjEyIDEzOjIzOjU5AP/hDM1odHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDQuNC4wLUV4aXYyIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0RXZ0PSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VFdmVudCMiIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1sbnM6R0lNUD0iaHR0cDovL3d3dy5naW1wLm9yZy94bXAvIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOkRvY3VtZW50SUQ9ImdpbXA6ZG9jaWQ6Z2ltcDpkNGYzODUwZC01ZDBhLTQ5YTMtYWE1Yy02MzE2YTU1ODQ1MzkiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6NTQ0ZDMxNDQtNTliNi00ZTE3LTg2NGItOTJkMmMyOTk2MTBlIiB4bXBNTTpPcmlnaW5hbERvY3VtZW50SUQ9InhtcC5kaWQ6MWNmMTExNjAtNjk2Ny00ODM0LWJhYmUtZTM2OGQzMmRiMzI0IiBkYzpGb3JtYXQ9ImltYWdlL2pwZWciIEdJTVA6QVBJPSIyLjAiIEdJTVA6UGxhdGZvcm09IldpbmRvd3MiIEdJTVA6VGltZVN0YW1wPSIxNzA3NzYyMjQ2NDM0OTQ2IiBHSU1QOlZlcnNpb249IjIuMTAuMzQiIHhtcDpDcmVhdG9yVG9vbD0iR0lNUCAyLjEwIiB4bXA6TWV0YWRhdGFEYXRlPSIyMDI0OjAyOjEyVDEzOjIzOjU5LTA1OjAwIiB4bXA6TW9kaWZ5RGF0ZT0iMjAyNDowMjoxMlQxMzoyMzo1OS0wNTowMCI+IDx4bXBNTTpIaXN0b3J5PiA8cmRmOlNlcT4gPHJkZjpsaSBzdEV2dDphY3Rpb249InNhdmVkIiBzdEV2dDpjaGFuZ2VkPSIvIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOmY4MGQ3NThjLTlmMmYtNDI4ZC1iMTI4LTdiMmY2ZTM5YTJkMyIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iR2ltcCAyLjEwIChXaW5kb3dzKSIgc3RFdnQ6d2hlbj0iMjAyNC0wMi0xMlQxMzoyNDowNiIvPiA8L3JkZjpTZXE+IDwveG1wTU06SGlzdG9yeT4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPD94cGFja2V0IGVuZD0idyI/Pv/iAhxJQ0NfUFJPRklMRQABAQAAAgxsY21zAhAAAG1udHJSR0IgWFlaIAfcAAEAGQADACkAOWFjc3BBUFBMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD21gABAAAAANMtbGNtcwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACmRlc2MAAAD8AAAAXmNwcnQAAAFcAAAAC3d0cHQAAAFoAAAAFGJrcHQAAAF8AAAAFHJYWVoAAAGQAAAAFGdYWVoAAAGkAAAAFGJYWVoAAAG4AAAAFHJUUkMAAAHMAAAAQGdUUkMAAAHMAAAAQGJUUkMAAAHMAAAAQGRlc2MAAAAAAAAAA2MyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHRleHQAAAAARkIAAFhZWiAAAAAAAAD21gABAAAAANMtWFlaIAAAAAAAAAMWAAADMwAAAqRYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9jdXJ2AAAAAAAAABoAAADLAckDYwWSCGsL9hA/FVEbNCHxKZAyGDuSRgVRd13ta3B6BYmxmnysab9908PpMP///9sAQwBQNzxGPDJQRkFGWlVQX3jIgnhubnj1r7mRyP///////////////////////////////////////////////////9sAQwFVWlp4aXjrgoLr/////////////////////////////////////////////////////////////////////////8IAEQgAVQBkAwERAAIRAQMRAf/EABcAAQEBAQAAAAAAAAAAAAAAAAABAgP/xAAVAQEBAAAAAAAAAAAAAAAAAAAAAf/aAAwDAQACEAMQAAABGQUpUAyRaQsKhpNEWAFBDJqJWkpCABRQZEWtIIChYkUUGY1RBQUBYQAhQgFABFAEKVAIaIpIoAkZrokIUpFJSKBIzWioAAAUQRmhoqAAFAQM1AClAICxoH//xAAYEAEAAwEAAAAAAAAAAAAAAAARACAwYP/aAAgBAQABBQKpCGB3Dh//xAAUEQEAAAAAAAAAAAAAAAAAAABw/9oACAEDAQE/AQ//xAAUEQEAAAAAAAAAAAAAAAAAAABw/9oACAECAQE/AQ//xAAUEAEAAAAAAAAAAAAAAAAAAABw/9oACAEBAAY/Ag//xAAdEAACAwEBAQEBAAAAAAAAAAABEQAQIDEwIUFx/9oACAEBAAE/IbUANP6oRpQRzLojAEPiRQ+nCtaHLeTg9g54uOjBy3Hg4PYCjp+AP5g6OAZ826WWaOOODH//2gAMAwEAAgADAAAAEAIA2/BMJ2JJIAw2bbIJFC3+vIJgW++sBIC//wDwSQPvveATR/t5sCMTt9uyTiB9syBUCCCSDm//xAAUEQEAAAAAAAAAAAAAAAAAAABw/9oACAEDAQE/EA//xAAWEQADAAAAAAAAAAAAAAAAAAABUGD/2gAIAQIBAT8QTio//8QAHxAAAgMAAwEBAQEAAAAAAAAAAAERITEQQVFhcSCB/9oACAEBAAE/EG6GKz4GjFtD+B8NDUpEhwLiKLFUtkrA3JJZLEKeoanVr0W2J0TZ3DD/ALR2oTaoT8Bs1ECYShZYr1EniI5Xno6ZshiFaxt+Ev3hzPQjghoSIPyMfvKQ4XVkSLR0OofpL0lxv/Df9Ek9IRDotqsTlMWk+/wRsNwFanhiG4RIk2ucwcyYKiQ64WmuUPRKSU+QW+zNFR0tShNqTFX8JPobScOxydDlqxTWj/RO+OxK7JdpC+UN14NnrKWK2NRx/9k="):
    #print(v)

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