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

import os
os.environ["OPENAI_API_KEY"] = KEYS['openai']
openai.api_key = KEYS['openai']


datafile_path = "data/concepts_embeddings.csv"
df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

# search through the reviews for a specific product
def search_concepts(df, product_description, n=3, pprint=False):
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

def getFromConceptsFile(file, query):
    loader = TextLoader("data/"+file)
    data = loader.load()

    embeddings = OpenAIEmbeddings(openai_api_key = openai.api_key)

    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=10)
    texts = text_splitter.split_documents(data)

    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0.0,model_name='gpt-4'), chain_type="stuff", retriever=retriever)
    results = qa.run(query)
    print(results)
    return json.loads(results)
    

def searchScientificNames(
    description: Optional[str] = None
) -> str:
    """Function to get a list of scientific names from the provided names that fit a description"""
    #query = 'Find at most 10 scientific names that fit the description "'+description+'". ONLY return a JSON list, and nothing more.'
    #return getFromConceptsFile('concepts1.txt', query) + getFromConceptsFile('concepts2.txt', query)
    
    results = search_concepts(df, description, n=10)
    results = results.values.tolist()
    if 'marine organism' in results:
        results.remove('marine organism')
    print(results)
    return json.dumps(results)


def initLangchain():
    searchScientificNames_tool = StructuredTool.from_function(
        searchScientificNames
        )
        
    chat = ChatOpenAI(model_name="gpt-4",temperature=0, openai_api_key = openai.api_key)
    tools = [searchScientificNames_tool]
    return initialize_agent(tools,
                               chat,
                               agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                               verbose=True)

def getScientificNamesLangchain(concept):
    try:
        data = agent_chain("Get me scientific names of "+concept+". Output a comma-separated list and nothing more.")
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
#print(searchScientificNames('tentacles'))
