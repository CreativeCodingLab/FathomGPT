from constants import *
from utils import getScientificNames

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
    """Function to get all scientific names that fits a description"""
    results = getScientificNames(description)
    candidates = getConceptCandidates(df, description)
    results.extend(candidates.values.tolist())
    results = list(dict.fromkeys(results))
    print(results)
    results = filterScientificNames(description, results)
    print(results)
    return results

def GetSQLResult(query:str):
    isJSON = False
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
    except:
        print("Error processing sql server response")
        output = query

    return (isJSON, output)


def generateSQLQuery(
    prompt: str
) -> (bool, str):
    """Converts text to sql. It is important to provide a scientific name when a species data is provided. If the data is need for specific task, input the task too. The database has image, bounding box and marine regions table. The datbase has data of species in a marine region with the corresponding images"""

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
    
    return sqlQuery.content

    


def initLangchain(messages=[]):
    
    memory = ConversationBufferMemory(memory_key="chat_history")

    for m in messages:
        memory.save_context({"input": m['prompt']}, {"input": m['response']})

    getScientificNamesFromDescription_tool = StructuredTool.from_function(
        getScientificNamesFromDescription,
        )
    generateSQLQuery_tool = StructuredTool.from_function(
        generateSQLQuery
        )
        
    chat = ChatOpenAI(model_name="gpt-4",temperature=0, openai_api_key = openai.api_key)
    tools = [getScientificNamesFromDescription_tool, generateSQLQuery_tool]


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

# messages must be in the format: [{"prompt": prompt, "response": json.dumps(response)}]
def get_Response(prompt, messages=[]):
    agent_chain = initLangchain(messages)
    
    if DEBUG_LEVEL >= 3:
        print(agent_chain)

    sql_query = agent_chain.run(input="Your function is to generate sql for the prompt using the tools provided. Output only the sql query. Prompt: "+prompt)

    isJSON, result = GetSQLResult(sql_query['output'])


    summerizerResponse = openai.ChatCompletion.create(
        model="gpt-4-0613",
        temperature=0,
        messages=[{"role": "system","content":"""Based on the below details output a json in provided format. The response must be a json. 
        
        {
            "outputType": "", //enum(image, text, table, heatmap, vegaLite) The data type based on the 'input'
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
    if(isJSON):
        output["species"] = result
    elif(summaryPromptResponse["outputType"]!="vegaSchema"):
        output["table"] = result

    return output





#DEBUG_LEVEL = 5
#print(agent_chain(getSciNamesPrompt('fused carapace'))['output'])
#print(getScientificNamesLangchain('rattail'))

#print(get_Response("Display a bar chart illustrating the distribution of every species in Monterey Bay, categorized by standard ocean depth intervals."))
#print(get_Response("Display a pie chart illustrating the distribution of every species in Monterey Bay, categorized by standard ocean depth intervals."))
#print(get_Response("Generate a heatmap of species in Monterey Bay"))
#print(get_Response("Show me images of Aurelia Aurita from Moneterey Bay"))
#print(get_Response("Find me images of species 'Aurelia aurita' in Monterey bay and depth less than 5k meters"))
#print(get_Response("What is the total number of images in the database?"))
#print(get_Response("What is the the most found species in the database and what is it's location?"))

test_msgs = [{"prompt": 'Find me images of Moon jellyfish', "response": json.dumps({'a': '123', 'b': '456'})}, {"prompt": 'What are they', "response": json.dumps({'responseText': 'They are creatures found in Lake Ontario'})}]
print(get_Response("Where can I find them", test_msgs))

