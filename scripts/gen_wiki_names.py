import pandas as pd

import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

import os
os.environ["OPENAI_API_KEY"] = 
openai.api_key = os.environ["OPENAI_API_KEY"]

def filterScientificNames(
    commonName: str,
    scientificNames: list
) -> str:
    template = """A user will pass in a common name, and you should select all objects from scientificNames that match the common name.
    ONLY return a comma separated list, and nothing more."""
    human_template = "{commonName} {scientificNames}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI(model_name="gpt-4-0613",temperature=0, openai_api_key = openai.api_key)
    data = chain.invoke({"commonName": commonName, "scientificNames": scientificNames})
    return data.content


names = {}
input_datapath = "data/concepts_desc.csv" 
df = pd.read_csv(input_datapath)
df = df[["concepts", "names", "links", "description", "related"]]
df = df.dropna()
for index, row in df.iterrows():
    cnames = []
    if row['names'] != ' ':
        cnames.extend(row['names'].split(', '))
    if row['links'] != ' ':
        cnames.extend(row['links'].split(', '))
    concept = row['concepts']
    for name in cnames:
        if name in names:
            if len(names[name]) > 30:
                continue
            if concept not in names[name]:
                names[name].append(concept)
        else:
            names[name] = [concept]
            
print(len(names)) # 4473
# running filterScientificNames will be too expensive
"""
for name in list(names)[:10]:
    print(name)
    print(names[name])
    print(filterScientificNames(name, names[name]))
"""
