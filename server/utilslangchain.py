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

from langchain.tools.base import StructuredTool
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
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


datafile_path = "data/concepts_embeddings.csv"
df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

# search through the reviews for a specific product
def search_concepts(df, product_description, n=LANGCHAIN_SEARCH_CONCEPTS_TOPN, pprint=False):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    df = df.sort_values("similarity", ascending=False)
    
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
    
    
def getAvailableScientificNames(
    description: str
) -> list:
    """Function to get all available scientific names that fit a description"""
    results = search_concepts(df, description)
    results = results.values.tolist()
    if 'marine organism' in results:
        results.remove('marine organism')
    print(results)
    return json.dumps(results)


def searchScientificNames(
    description: Optional[str] = None,
    names: Optional[list] = None
) -> str:
    """Function to get a list of scientific names from the provided names that fit a description"""
    template = """You generate comma separated lists.
    A user will pass in a description, and you should select all objects from names that fit the description.
    ONLY return a comma separated list, and nothing more."""
    human_template = "{description} {names}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI(model_name="gpt-4-0613",temperature=0, openai_api_key = openai.api_key)
    data = chain.invoke({"description": description, "names": names})
    return data


def initLangchain():
    searchScientificNames_tool = StructuredTool.from_function(
        searchScientificNames
        )
    getAvailableScientificNames_tool = StructuredTool.from_function(
        getAvailableScientificNames
        )
        
    chat = ChatOpenAI(model_name="gpt-4",temperature=0, openai_api_key = openai.api_key)
    tools = [getAvailableScientificNames_tool, searchScientificNames_tool]
    return initialize_agent(tools,
                               chat,
                               agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                               verbose=(DEBUG_LEVEL >= 1))
                               
def getSciNamesPrompt(concept):
    template = """First get all the available scientific names that fit the description. Then search names that fit the description. ONLY return a machine-readable JSON list, and nothing more."""
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

agent_chain = initLangchain()

print(agent_chain(getSciNamesPrompt('tentacles'))['output'])

#print(searchScientificNames('tentacles'))
