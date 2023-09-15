from constants import *

import openai
import json
from datetime import datetime
import math
from urllib.request import urlopen
from urllib.parse import quote
from typing import Optional

from langchain.tools.base import StructuredTool
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import initialize_agent, AgentType

openai.api_key = KEYS['openai']

def getScientificNameFromBiologicalConcept(
    concept: Optional[str] = None,
) -> str:
    """"Function to generate scientific name from common name"""
    newChat = ChatOpenAI(model_name="gpt-4",temperature=0, openai_api_key = openai.api_key)
    data = newChat([
        HumanMessage(
          content="Get me scientific names of "+concept+" and output a JSON list."
        )
      ]
    )
    return data

def initLangchain():
    getScientificName_tool = StructuredTool.from_function(
        getScientificNameFromBiologicalConcept
        )
        
    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613",temperature=0, openai_api_key = openai.api_key)
    tools = [getScientificName_tool]
    return initialize_agent(tools,
                               chat,
                               agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                               verbose=True)
agent_chain = initLangchain()

def getScientificNamesLangchain(concept):
    try:
        data = agent_chain("Get me scientific names of "+concept+" and output a JSON list.")
        data = data['output']
        if DEBUG_LEVEL >= 1:
            print('Fetched scientific names from Langchain:')
            print(data)
        data = json.loads(data)
        data = [d['ScientificName'] for d in data if 'ScientificName' in d]
        return data
    except:
        if DEBUG_LEVEL >= 1:
            print('failed to fetch Langchain scientific names')
        return []
