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

def modifyExistingChart(prompt: str, schema: str):
    summerizerResponse = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            temperature=0,
            messages=[{"role": "system","content":"""
            Modify the vega lite chart schema based on the instruction provided. Output only the chartSchema

            """},{"role":"user", "content": "{\"chartSchema\": \"" + json.dumps(schema) + "\", \"\ninstruction\":\"" + prompt+ "\"}"}],
        )
    

    return summerizerResponse["choices"][0]["message"]["content"]

    
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
        return ", ".join(results)
    
    return "anything"


def getAnswer(
    question: str,
) -> list:
    """Function for questions about the features of a species.
    DO NOT use this tool for fetching images, taxonomy or generating charts"""
    print("question: "+question)

    """ response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role":"system","content":"answer the user's question"},{"role":"user","content":question}],
            function_call="auto",
            temperature=0,

        )
    return response["choices"][0]["message"] """
    return ""
    

# ==== SQL database query ====

def GetSQLResult(query: str, isVisualization: bool = False):
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
        print(query)

        cursor.execute(query)

        rows = cursor.fetchall()

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
            return (False, results_dict)

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
                    row_data[columns[idx]] = value
                output.append(row_data)
            isSpeciesData = True
        else:
            print("No rows found for this query.")
            output = None
    except Exception as e:
        print("Error processing sql server response", e)
        output = "Error"

    return (isSpeciesData, output)

oneShotData = {
    "images": {
        "instructions": "The sql query must have bounding box id of the species, concept of the species and the image url of the species on all inputs",
        "user": """
            User Prompt: "Provide me few images of Asterias rubens, Acanthaster planci, Linckia laevigata, Protoreaster nodosus, Pycnopodia helianthoides"
            Output type: images
            InputImageDataAvailable: False""",
        "assistant": """
        {
            "sqlQuery": "SELECT TOP 10     i.url AS url,     b.concept AS concept,     b.id as id,     b.image_id as image_id FROM      dbo.bounding_boxes AS b JOIN      dbo.images AS i ON b.image_id = i.id WHERE      b.concept IN ('Asterias rubens', 'Acanthaster planci', 'Linckia laevigata', 'Protoreaster nodosus', 'Pycnopodia helianthoides')",
            "responseText": "Few images of Asterias rubens, Acanthaster planci, Linckia laevigata, Protoreaster nodosus, Pycnopodia helianthoides are shown below."
        }""",
        "user2": """
            User Prompt: "Find me images of Aurelia aurita that looks good"
            Output type: images
            InputImageDataAvailable: False""",
        "assistant2": """
        {
            "sqlQuery": "SELECT TOP 10 i.url, b.concept, b.id, i.id as image_id FROM dbo.bounding_boxes AS b JOIN dbo.images AS i ON b.image_id = i.id WHERE b.concept = 'Aurelia aurita' ORDER BY (b.width * b.height) / (i.width * i.height) DESC",
            "responseText": "Here is an image of Aurelia aurita that you may find appealing."
        }""",
    },
    "text": {
        "instructions": "Make sure the response text is a templated string so that data can be formatted inside the text",
        "user": f"""
            User Prompt: ""How many images of Pycnopodia helianthoides are in the database"
            Output type: text
            InputImageDataAvailable: False""",
        "assistant": """{
                "sqlQuery": "SELECT COUNT(*) as TotalImages FROM dbo.bounding_boxes  WHERE concept = 'Pycnopodia helianthoides'",
                "responseText": "There are {TotalImages} images of Pycnopodia helianthoides in the database."
            }""",
        "user2": f"""
            User Prompt: ""In what pressure level is the species with bounding box id 2258729 living at"
            Output type: text
            InputImageDataAvailable: False""",
        "assistant2": """{
                "sqlQuery": "SELECT images.pressure_dbar, bounding_boxes.concept FROM dbo.bounding_boxes JOIN dbo.images ON bounding_boxes.image_id = images.id WHERE bounding_boxes.id = 2258739;",
                "responseText": "The species with bounding box id 2258729 us {concept}. It is living at pressure {pressure_dbar} dbar."
            }""",
    },
    "imagesWithInput": {
        "instructions": "In the SQL query, there should be a specific spot where the feature vector of the input image can be inserted, denoted as '{imageFeatureVector}'. This allows the SQL query to use the data from the input image's features. If the data for the image isn't provided (InputImageDataAvailable is false), then the response text should clearly say that the input image data is missing.",
        "user": f"""
            User Prompt: "Find me similar images of species that are not Bathochordaeus stygius"
            Output type: "images"
            InputImageDataAvailable: True""",
        "assistant": """"
        {
            "sqlQuery": "IF OBJECT_ID('tempdb..#InputFeatureVectors') IS NOT NULL DROP TABLE #InputFeatureVectors; CREATE TABLE #InputFeatureVectors ( vector_index INT, vector_value DECIMAL(18,5) ); INSERT INTO #InputFeatureVectors (vector_index, vector_value) VALUES {imageFeatureVector}; WITH FeatureVectors AS ( SELECT BBFV.bounding_box_id AS BoxID, SUM(IFV.vector_value * BBFV.vector_value) AS DotProduct FROM #InputFeatureVectors IFV INNER JOIN bounding_box_image_feature_vectors BBFV ON IFV.vector_index = BBFV.vector_index GROUP BY BBFV.bounding_box_id ), Magnitude AS ( SELECT BB.id AS BoxID, BB.magnitude FROM bounding_boxes BB WHERE concept = 'Bathochordaeus stygius' ), InputMagnitude AS ( SELECT SQRT(SUM(POWER(vector_value, 2))) AS magnitude FROM #InputFeatureVectors ), SimilaritySearch AS ( SELECT TOP 10 FV.BoxID as id, FV.DotProduct / (IM.magnitude * M.magnitude) AS CosineSimilarity FROM FeatureVectors FV CROSS JOIN InputMagnitude IM INNER JOIN Magnitude M ON FV.BoxID = M.BoxID WHERE IM.magnitude IS NOT NULL AND M.magnitude IS NOT NULL AND IM.magnitude > 0 AND M.magnitude > 0 ORDER BY CosineSimilarity DESC ) SELECT SS.id AS id, BB.concept, IMG.url, SS.CosineSimilarity FROM SimilaritySearch SS INNER JOIN bounding_boxes BB ON BB.id = SS.id INNER JOIN images IMG ON BB.image_id = IMG.id; DROP TABLE #InputFeatureVectors;",
            "responseText": "Here are the similar images of species that are not Bathochordaeus stygius."
        }""",
        "user2": f"""
            User Prompt: "Find species simiar to the one found in image id 2553888"
            Output type: "images"
            InputImageDataAvailable: True""",
        "assistant2": """"
        {
            "sqlQuery": "DECLARE @InputImageID INT; SET @InputImageID = 2553888; WITH BoundingBoxes AS ( SELECT id AS BoxID, image_id FROM bounding_boxes WHERE image_id = @InputImageID ), FeatureVectors AS ( SELECT BB.BoxID, BBFV.vector_index, BBFV.vector_value FROM BoundingBoxes BB INNER JOIN bounding_box_image_feature_vectors BBFV ON BB.BoxID = BBFV.bounding_box_id ), Magnitude AS ( SELECT BB.BoxID, SQRT(SUM(POWER(BBFV.vector_value, 2))) AS magnitude FROM BoundingBoxes BB INNER JOIN bounding_box_image_feature_vectors BBFV ON BB.BoxID = BBFV.bounding_box_id GROUP BY BB.BoxID ), SimilaritySearch AS ( SELECT FV1.BoxID AS BoxID1, FV2.BoxID AS BoxID2, SUM(FV1.vector_value * FV2.vector_value) / (M1.magnitude * M2.magnitude) AS CosineSimilarity FROM FeatureVectors FV1 INNER JOIN FeatureVectors FV2 ON FV1.vector_index = FV2.vector_index AND FV1.BoxID < FV2.BoxID INNER JOIN Magnitude M1 ON FV1.BoxID = M1.BoxID INNER JOIN Magnitude M2 ON FV2.BoxID = M2.BoxID GROUP BY FV1.BoxID, FV2.BoxID, M1.magnitude, M2.magnitude ), TopSimilarity AS ( SELECT TOP 10 SS.BoxID1, SS.BoxID2, SS.CosineSimilarity FROM SimilaritySearch SS ORDER BY SS.CosineSimilarity DESC ) SELECT TS.BoxID1, TS.BoxID2, BB1.concept AS Concept1, BB2.concept AS Concept2, IMG.url, TS.CosineSimilarity FROM TopSimilarity TS INNER JOIN bounding_boxes BB1 ON BB1.id = TS.BoxID1 INNER JOIN bounding_boxes BB2 ON BB2.id = TS.BoxID2 INNER JOIN images IMG ON BB1.image_id = IMG.id;",
            "responseText": "Here are the images of species similar to the species found in image id 2553888."
        }""",
    },
    "visualization": {
        "instructions": "Generate sample data and corresponding python Plotly code.Guarantee that the produced SQL query and Plotly code are free of syntax errors and do not contain comments.In the Plotly code, ensure all double quotation marks ("") are properly escaped with a backslash ().Represent newline characters as \\n and tab characters as \\t within the Plotly code. The input data object is just a list of object, if you want it to be pandas data frame object, convert it first. Donot use mapbox, use openstreet maps instead.",
        "user": f"""
            User Prompt: "Create a network visualization showing the species co-occurence of species within an image."
            Output type: visualization
            InputImageDataAvailable: False""",
        "assistant": """
        {
            "sqlQuery": "SELECT bb1.concept AS Concept1, bb2.concept AS Concept2 FROM dbo.bounding_boxes bb1 JOIN dbo.bounding_boxes bb2 ON bb1.image_id = bb2.image_id AND bb1.concept <> bb2.concept;",
            "sampleData": "{'Concept1':['Species 1', 'Species 2'], 'Concept2':['Species 3', Species 4']}",
            "plotlyCode": "import pandas as pd\nimport plotly.graph_objs as go\nimport networkx as nx\ndef drawVisualization(data):\n\tdf = pd.DataFrame(data)\n\tG = nx.Graph()\n\tfor i, row in df.iterrows():\n\t\tG.add_edge(row['Concept1'], row['Concept2'])\n\tpos = nx.kamada_kawai_layout(G)\n\tedge_x = []\n\tedge_y = []\n\tfor edge in G.edges():\n\t\tx0, y0 = pos[edge[0]]\n\t\tx1, y1 = pos[edge[1]]\n\t\tedge_x.extend([x0, x1, None])\n\t\tedge_y.extend([y0, y1, None])\n\tedge_trace = go.Scatter(\n\t\tx=edge_x, y=edge_y,\n\t\tline=dict(width=0.5, color='#888'),\n\t\thoverinfo='none',\n\t\tmode='lines')\n\tnode_x = []\n\tnode_y = []\n\tfor node in G.nodes():\n\t\tx, y = pos[node]\n\t\tnode_x.append(x)\n\t\tnode_y.append(y)\n\tnode_trace = go.Scatter(\n\t\tx=node_x, y=node_y,\n\t\tmode='markers',\n\t\thoverinfo='text',\n\t\tmarker=dict(\n\t\t\tshowscale=True,\n\t\t\t\t  colorscale='YlGnBu',\n\t\t\tsize=[len(list(G.neighbors(node))) * 10 for node in G.nodes()],\n\t\t\tcolor=[len(list(G.neighbors(node))) for node in G.nodes()],\n\t\t\tcolorbar=dict(\n\t\t\t\tthickness=15,\n\t\t\t\ttitle='Node Connections',\n\t\t\t\txanchor='left',\n\t\t\t\ttitleside='right'\n\t\t\t),\n\t\t\tline_width=2))\n\tnode_trace.text = [f'{node} ({len(list(G.neighbors(node)))} connections)' for node in G.nodes()]\n\tfig = go.Figure(data=[edge_trace, node_trace],\n\t\t\t\tlayout=go.Layout(\n\t\t\t\t\ttitle='<br>Network graph of species co-occurrence',\n\t\t\t\t\ttitlefont_size=16,\n\t\t\t\t\tshowlegend=False,\n\t\t\t\t\thovermode='closest',\n\t\t\t\t\tmargin=dict(b=20,l=5,r=5,t=40),\n\t\t\t\t\txaxis=dict(showgrid=False, zeroline=False, showticklabels=False),\n\t\t\t\t\tyaxis=dict(showgrid=False, zeroline=False, showticklabels=False))\n\t\t\t\t\t)\n\treturn fig",
            "responseText": "Below is the network visualization showing the species co-occurence of species within an image."
        }""",
        "user2": f"""
            User Prompt: "Generate an Interactive Time-lapse Map of Marine Species Observations Grouped by Year"
            Output type: visualization
            InputImageDataAvailable: False""",
        "assistant2": """
        {
            "sqlQuery": "SELECT      bb.concept,     i.latitude,      i.longitude,      YEAR(i.timestamp) AS ObservationYear FROM      dbo.bounding_boxes bb JOIN dbo.images i ON bb.image_id = i.id WHERE      i.latitude IS NOT NULL AND      i.longitude IS NOT NULL AND     i.timestamp IS NOT NULL ORDER BY      i.timestamp;",
            "sampleData": "{     'concept': ['Species A', 'Species B', 'Species A', 'Species B'],     'latitude': [35.6895, 35.6895, 35.6895, 35.6895],     'longitude': [139.6917, 139.6917, 139.6917, 139.6917],  'ObservationYear': [2020, 2021, 2022, 2023]  }",
            "plotlyCode": "import plotly.graph_objs as go\nfrom plotly.subplots import make_subplots\ndef drawVisualization(data):\n\tfig = go.Figure()\n\tfor year in sorted(set(data['ObservationYear'])):\n\t\tfig.add_trace(\n\t\t\tgo.Scattergeo(\n\t\t\t\tlon=[data['longitude'][i] for i in range(len(data['longitude'])) if data['ObservationYear'][i] == year],\n\t\t\t\tlat=[data['latitude'][i] for i in range(len(data['latitude'])) if data['ObservationYear'][i] == year],\n\t\t\t\ttext=[f\"{data['concept'][i]} ({year})\" for i in range(len(data['concept'])) if data['ObservationYear'][i] == year],\n\t\t\t\tmode='markers',\n\t\t\t\tmarker=dict(\n\t\t\t\t\tsize=8,\n\t\t\t\t\tsymbol='circle',\n\t\t\t\t\tline=dict(width=1, color='rgba(102, 102, 102)')\n\t\t\t\t),\n\t\t\t\tname=f\"Year {year}\",\n\t\t\t\tvisible=(year == min(data['ObservationYear'])) \n\t\t\t)\n\t\t)\n\tsteps = []\n\tfor i, year in enumerate(sorted(set(data['ObservationYear']))):\n\t\tstep = dict(\n\t\t\tmethod='update',\n\t\t\targs=[{'visible': [False] * len(fig.data)},\n\t\t\t\t{'title': f\"Observations for Year: {year}\"}], \n\t\t)\n\t\tstep['args'][0]['visible'][i] = True   i'th trace to \"visible\"\n\t\tsteps.append(step)\n\tsliders = [dict(\n\t\tactive=10,\n\t\tcurrentvalue={\"prefix\": \"Year: \"},\n\t\tpad={\"t\": 50},\n\t\tsteps=steps\n\t)]\n\tfig.update_layout(\n\t\tsliders=sliders,\n\t\ttitle='Time-lapse Visualization of Species Observations',\n\t\tgeo=dict(\n\t\t\tscope='world',\n\t\t\tprojection_type='equirectangular',\n\t\t\tshowland=True,\n\t\t\tlandcolor='rgb(243, 243, 243)',\n\t\t\tcountrycolor='rgb(204, 204, 204)',\n\t\t),\n\t)\n\treturn fig",
            "responseText": "Below is an Interactive Time-lapse Map of Marine Species Observations Grouped by Year."
        }""",
    },
    "table": {
        "instructions": "The response text can be templated so that it can hold the count of the data array from the sql query result.",
        "user": f"""
            User Prompt: "List the species that are found in image with id 2256720"
            Output type: table
            InputImageDataAvailable: False""",
        "user2": f"""
            User Prompt: "What species are frequently found at 1000m depth?"
            Output type: table
            InputImageDataAvailable: False""",
        "assistant": """
        {
            "sqlQuery": "SELECT b.concept FROM dbo.bounding_boxes AS b JOIN dbo.images AS i ON b.image_id = i.id WHERE b.image_id = 2256720;",
            "responseText": "The table below lists all the species found in image with id 2256720."
        }""",
        "assistant2": """
        {
            "sqlQuery": "SELECT b.concept AS species, COUNT(*) AS frequency FROM dbo.bounding_boxes AS b JOIN dbo.images AS i ON b.image_id = i.id WHERE i.depth_meters = 1000 GROUP BY b.concept ORDER BY frequency DESC;",
            "responseText": "Table shows the frequently found species at 1000m depth and their count."
        }""",
    },
}
def generateSQLQuery(
    prompt: str,
    scientificNames: str,
    name: str,
    outputType: str,
    inputImageDataAvailable: bool
) -> str:
    """Converts text to sql. If the common name of a species is provided, it is important convert it to its scientific name. If the data is need for specific task, input the task too. The database has image, bounding box and marine regions table. The database has data of species in a marine region with the corresponding images.
    """
    
    if len(name) > 1:
        prompt = prompt.replace(name, scientificNames)
    elif len(scientificNames) > 1:
        prompt = prompt.rstrip(".")+" with names: "+scientificNames

    print("------------------------prompt passed "+ prompt)

    completion = openai.ChatCompletion.create(
        model='gpt-4-turbo-preview', #if outputType == "visualization" else SQL_FINE_TUNED_MODEL,
        messages=[
            {"role": "system", "content": """You are a very intelligent json generated that can generate highly efficient sql queries and plotly python code. You will be given an input prompt for which you need to generated the JSON in a format given below, nothing else.
                The Generated SQL must be valid
                The JSON format and the attributes on the JSON are provided below
                {
                    "sqlQuery": "",
                    "sampleData": "",
                    "plotlyCode": "",
                    "responseText": ""
                }

                sqlQuery: This is the sql server query you need to generate based on the user's prompt. The database structure provided will be very useful to generate the sql query. 
                sampleData: This is the sample data that you think should be generated when running the sql query. This is optional. It is only needed when the outputType is visualization
                plotlyCode: This is the python plotly code that you will generate. You will generate a function named "drawVisualization(data)". The function should take in data variable, which is a python list. The data value will have the structur of the sampleData generated above. Donot redfine the sample data here. The code should have the necessary imports and the "drawVisualization" function. This attribute is optional but must be generated only when the outputType is visualization.
                responseText: Suppose you are answering the user with the output from the prompt. You need to write the message in this section. When the response is text, you need to output the textResponse in a way the values from the generated sql can be formatted in the text



                SQL Server Database Structure:
             """+DB_STRUCTURE+"\n\n"+ (oneShotData['imagesWithInput']['instructions'] if (inputImageDataAvailable == True) else "") + "\n"+
                oneShotData[outputType]['instructions']
            },
            {
                "role": "user","content": oneShotData["imagesWithInput" if inputImageDataAvailable else outputType]['user']
            },
            {
                "role": "assistant","content": oneShotData["imagesWithInput" if inputImageDataAvailable else outputType]['assistant']
            },
            {
                "role": "user","content": oneShotData["imagesWithInput" if inputImageDataAvailable else outputType]['user2']
            },
            {
                "role": "assistant","content": oneShotData["imagesWithInput" if inputImageDataAvailable else outputType]['assistant2']
            },
            {
                "role": "user","content": f"""
                User Prompt: {prompt}
                Output type: {outputType}
                InputImageDataAvailable: {inputImageDataAvailable}"""
            },
        ],
        temperature=0
    )

    result = completion.choices[0].message.content
    first_brace_position = result.find('{')
    if first_brace_position!=0:
        result = result[first_brace_position:]
    if result.endswith('```'):
        result = result[:-3]
    result = json.loads(result)
    result['outputType'] = outputType
    return result



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
            "outputType": {
                "type": "string",
                "description": "enum of 'images', 'text', 'visualization','table'. The enum type must be deduced based on the prompt. If the prompt says find me number of images or name of the species, etc. or if the prompt can be answered in a single sentence without any images or chart or multi-dimensional data shown, outputType is a text. If the prompt says to generate or find image, outputType is image, if prompt says to generate any kind of visualization liek graph, plots,etc., the outputType is visualization. If nothing matches, it is a table"
            }
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
#{
#    "name": "modifyExistingChart",
#    "description": "Important:Donot run this function when the prompt says to generate a chart. This tool that modifies the chart schema based on the provided schema. This tool only makes visual changes to the chart. If the data itself needs to be changed, do not use this tool.",
#    "parameters": {
#        "type": "object",
#        "properties": {
#            "prompt": {
#                "type": "string",
#                "description": "Prompt that instructs what modification that needs to be done in the chart",
#            },
#        },
#        "required": ["prompt"],
#    },
#}
]

availableFunctionsDescription = {
    "getScientificNamesFromDescription": "Generating scientific name from description",
    
    "generateSQLQuery": "Generating SQL Query",
    
    "getTaxonomyTree": "Getting the taxonomy tree",
    
    "getTaxonomicRelatives": "Getting taxonomic relatives",
    
    "getAnswer": "Getting answer from ChatGPT.",

    "modifyExistingChart": "Modifying the chart.",

}

# messages must be in the format: [{"prompt": prompt, "response": json.dumps(response)}]
def get_Response(prompt, imageData="", messages=[], isEventStream=False, db_obj=None):

#    code_string = """
#import pandas as pdimport plotly.express as pxdef drawVisualization(data):    df = pd.DataFrame(data)    fig = px.density_mapbox(df, lat='latitude', lon='longitude', z='depth_meters', radius=10,                            center=dict(lat=36.6002, lon=-121.8947), zoom=10,                            mapbox_style="open-street-map",                            hover_data={'concept': True, 'depth_meters': False})  # Show 'concept' and hide 'depth_meters' on hover    fig.update_layout(        title='Heatmap of Species in Monterey Bay'    )    return fig
#    """
#
#    isSepceisData, data = GetSQLResult("SELECT     b.concept,     b.id,     i.latitude,     i.longitude,     i.depth_meters FROM     dbo.bounding_boxes b     JOIN dbo.images i ON b.image_id = i.id     JOIN dbo.marine_regions mr ON i.latitude BETWEEN mr.min_latitude AND mr.max_latitude         AND i.longitude BETWEEN mr.min_longitude AND mr.max_longitude WHERE     mr.name = 'Monterey Bay'", True)
#    exec(code_string, globals())
#
#    print("here")
#
#    fig = drawVisualization(data)
#    html_output = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
#    rs = {
#        "outputType": "visualization",
#        "html": html_output,
#        "responseText": "Her is your visualization"
#    }
#
#    event_data = {
#        "result": rs
#    }
#    sse_data = f"data: {json.dumps(event_data)}\n\n"
#    yield sse_data
#    return



    start_time = time.time()
    startingMessage = None
    if len(messages)!=0:
        startingMessage = json.loads(json.dumps(messages[len(messages)-1]))
    for smessage in messages:
        if(smessage["role"]=="assistant"):
            if(len(smessage["content"])>200):
                smessage["content"]=smessage["content"][:200]+"...\n"
    messages.append({"role":"user","content":"Use the tools provided to generate response to the prompt. Important: If the prompt contains a common name or description use the 'getScientificNamesFromDescription' tool first. The prompt might have refernce to previous prompts but the tools do not have previous memory. So do not use words like their, its in the input to the tools, provide the name. Prompt:"+prompt})
    isSpeciesData = False
    result = None
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
            formatted_feature = f"({index}, {feature:.5f})"
            formatted_features.append(formatted_feature)

        eval_image_feature_string = ", ".join(formatted_features)
    
    while curLoopCount < 4:
        curLoopCount+=1
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=availableFunctions,
            function_call="auto",
            temperature=0,

        )
        sqlQuery=None
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
            if(function_name=="modifyExistingChart"):
                if isEventStream:
                    event_data = {
                        "message": "Modifying the chart"
                    }
                    sse_data = f"data: {json.dumps(event_data)}\n\n"
                    yield sse_data

                evaluatedContent = eval(startingMessage['content'])
                prvSchema = evaluatedContent['vegaSchema']
                prvData = prvSchema['data']
                prvSchema['data']=None
                result = function_to_call(**args, schema=prvSchema)
                prvSchema = json.loads(result)
                prvSchema['data'] = prvData
                evaluatedContent['vegaSchema'] = prvSchema
                output = evaluatedContent

                if isEventStream:
                    Interaction.objects.create(main_object=db_obj, request=prompt, response=output)
                    output['guid']=str(db_obj.id)
                    event_data = {
                        "result": output
                    }
                    sse_data = f"data: {json.dumps(event_data)}\n\n"
                    yield sse_data
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
                if(function_name=="generateSQLQuery"):
                    args["inputImageDataAvailable"] = len(eval_image_feature_string)!=0
                result = function_to_call(**args)
                


                if(function_name=="generateSQLQuery"):
                    if len(eval_image_feature_string) != 0:
                        result['sqlQuery'] = result['sqlQuery'].replace("{imageFeatureVector}", eval_image_feature_string+";")
                    
                    sql = result['sqlQuery']
                    limit = -1
                    if sql.strip().startswith('SELECT '):
                        limit, sql = changeNumberToFetch(sql)
                    if isEventStream:
                        event_data = {
                            "message": "Querying database...                             SQL Query:"+sql
                        }
                        sse_data = f"data: {json.dumps(event_data)}\n\n"
                        yield sse_data
                    isSpeciesData, sqlResult = GetSQLResult(sql, result["outputType"]=="visualization")
                    if sqlResult == "Error":
                        event_data = {
                                "result": {
                                    "outputType": "error",
                                    "responseText": "Error running the generated sql query"
                                }
                            }
                        sse_data = f"data: {json.dumps(event_data)}\n\n"
                        yield sse_data
                        return None

                        
                    print("got data from db")

                    try:
                        sqlResult, isSpeciesData = postprocess(sqlResult, limit, prompt, sql, isSpeciesData)
                    except:
                        print('postprocessing error')
                        pass

                    if sqlResult and limit != -1 and limit < len(sqlResult) and isinstance(sqlResult, list):
                        sqlResult  = sqlResult[:limit]
                    print('isSpeciesData: '+str(isSpeciesData))

                    if(result["outputType"]=="visualization"):
                        exec(result["plotlyCode"], globals())
                    
                        fig = drawVisualization(sqlResult)
                        html_output = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                        result["html"]=html_output
                        result["species"]=sqlResult

                    elif(result["outputType"]=="table"):
                        result["table"]=sqlResult
                        result["responseText"]= result["responseText"].format(**sqlResult[0])
                    elif(result["outputType"]=="text"):
                        result["responseText"]= result["responseText"].format(**sqlResult[0])
                    else:
                        result["species"]=sqlResult
                        result["responseText"]= result["responseText"].format(**sqlResult[0])

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
            break

    output = {
        "outputType": result["outputType"],
        "responseText": result["responseText"] if result["responseText"] else "",
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
        Interaction.objects.create(main_object=db_obj, request=prompt, response=output)
        output['guid']=str(db_obj.id)
        event_data = {
            "result": output
        }
        sse_data = f"data: {json.dumps(event_data)}\n\n"
        yield sse_data
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
#    print(v)

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