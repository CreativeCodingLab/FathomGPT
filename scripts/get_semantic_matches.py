import json
import re
import pandas as pd
import numpy as np
import wikipedia

import openai
from ast import literal_eval
from openai.embeddings_utils import get_embedding, cosine_similarity
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate


openai.api_key = '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'

def getConceptCandidates(product_description, n=30, pprint=False):
    df = pd.read_csv("data/concepts_names_desc_embeddings.csv")
    df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    df = df.sort_values("similarity", ascending=False)
    df = df[df['similarity'] > 0.8] 
    
    #print(df.head(n))

    results = {}
    for index, row in df.head(n).iterrows():
        results[row['concepts']] = row['similarity']
    #print(results)
    return results

def filterScientificNames(
    description,
    names
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


def is_ascii(s):
    return all(ord(c) < 128 for c in s)
    
def isBiological(term):
    if term.count(' ') == 0 and (term[-2:] in ['ae', 'ea', 'us', 'ia', 'da', 'is'] or term[-4:] == 'form' or term[-6:] == 'formes' or term[-1:] == '/' or not is_ascii(term)):
        return True


    

terms = []
input_datapath = "data/concepts_desc.csv" 
df = pd.read_csv(input_datapath)
df = df[["concepts", "names", "links", "description", "related"]]
df = df.dropna()
for index, row in df.iterrows():
    if row['links'] != ' ':
        terms.extend(row['links'].split(', ')) 
terms = list(set(terms))
terms = [t for t in terms if not isBiological(t)]

f = open('data/semantic_matches.json')
semantic_scores = json.load(f)
semantic_scores = {s:t for (s, t) in semantic_scores.items() if not isBiological(s)}
terms = [t for t in terms if t not in semantic_scores]
print(len(terms))

c = 0
for kw in terms:
    print(str(c)+" "+kw)
    c = c + 1
    if kw not in semantic_scores:
        candidates = getConceptCandidates(kw)
        if len(candidates) == 0:
            continue
        """
        filtered = filterScientificNames(kw, list(candidates.keys()))
        print(candidates)
        if len(filtered) > 0:
            filtered = filtered.split(', ')
            candidates = {k:v for (k,v) in candidates.items() if k in filtered}
        print(candidates)
        """
        semantic_scores[kw] = candidates
    
    if c % 10 == 1:
        with open("data/semantic_matches.json", "w") as outfile:
            json.dump(semantic_scores, outfile)
    

