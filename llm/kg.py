from .constants import *

import openai
import json
from datetime import datetime
import math
from urllib.request import urlopen
from urllib.parse import quote

os.environ["OPENAI_API_KEY"] = KEYS['openai']
openai.api_key = KEYS['openai']

from fathomnet.api import images
from fathomnet.api import regions
from fathomnet.api import taxa
from fathomnet.api import boundingboxes
from fathomnet.dto import GeoImageConstraints

import time
import numpy as np
import os.path
from nltk.stem import PorterStemmer


def callGPT(instructions, text):
    try:
        answer = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-0125",
          timeout=30,
          messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text}
          ]
        )
        answer = json.loads(answer['choices'][0]['message']['content'].replace('```json\n', '').replace('```', ''))
    except:
        try:
            answer = openai.ChatCompletion.create(
              model="gpt-3.5-turbo-0125",
              timeout=30,
              messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": text}
              ]
            )
            answer = json.loads(answer['choices'][0]['message']['content'].replace('```json\n', '').replace('```', ''))
        except:
            return None
    
    return answer
    
    
def embed(terms):
    terms = [str(t) for t in terms if str(t) != '']
    
    resp = openai.Embedding.create(
        input=terms,
        engine="text-embedding-ada-002")

    embeddings = [r['embedding'] for r in resp['data']]
    return dict(zip(terms, embeddings))
    

    
def kg_name_res(prompt, instructions):
    

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        timeout=30,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt}
        ]
    )
    prompt_kg = callGPT(instructions, prompt)
    
    ps = PorterStemmer()
    
    if prompt_kg and 'subject' in prompt_kg and 'relation' in prompt_kg and 'object' in prompt_kg:
        prompt = {"s": ps.stem(prompt_kg['subject']), "o": ps.stem(prompt_kg['object']), "r": prompt_kg['relation'].lower()}
    else:
        return None
    print(prompt)

    prompt_embedding = openai.Embedding.create(
        input=prompt['r'],
        engine="text-embedding-ada-002")
    prompt_embedding = prompt_embedding['data'][0]['embedding']
    
    with open('scripts/kg/relations.json') as f:
        relations = json.load(f)
    
    
    
    rel = None
    for r in relations:
        sim = np.dot(relations[r], prompt_embedding)
        #print(r, sim)
        if sim > 0.9:
            rel = r
            break

    if not rel:
        return None
    
    with open('scripts/kg/kg_stemmed.json') as f:
        kgs = json.load(f)
        
    with open('scripts/kg/kg.json') as f:
        kg_raw = json.load(f)

    results = {}
    species_rel = {
        'predators': 'eats', 
        'eats': 'predators',
    }
    if prompt['r'] in species_rel:
        for c in kgs:
            kg = kgs[c]
            
            if prompt['s'] == c.lower(): 
                results.update(dict.fromkeys(kg_raw[c][rel], 0.75))
                #print(c, kg['alias'])
            elif prompt['o'] == c.lower():
                results.update(dict.fromkeys(kg_raw[c][species_rel[rel]], 0.75))
            
            if 'alias' in kg:
                if prompt['s'] in kg['alias']:
                    results.update(dict.fromkeys(kg_raw[c][rel], 0.6))
                elif prompt['o'] in kg['alias']:
                    results.update(dict.fromkeys(kg_raw[c][species_rel[rel]], 0.6))
                    
        concepts = boundingboxes.find_concepts()
        concepts = [c for c in concepts if c!='']
        sciNames = {}
        for r in results:
            #print(r)
            found = False
            for c in concepts:
                if '('+c.lower()+')' in r:
                    sciNames[c] = results[r] + 0.25
                elif ' ' in c and c.lower() in r:
                    sciNames[c] = results[r] + 0.25
        if len(sciNames) == 0:
            for r in results:
                if c.lower() in r:
                    sciNames[c] = results[r] + 0.2
        if len(sciNames) == 0:
            for r in results:
                for c in kg_raw:
                    if 'alias' in kg_raw[c] and r in kg_raw[c]['alias']:
                        sciNames[c] = results[r]
        results = sciNames
    
    if prompt['o'] != 'unknown' and len(results) == 0:
        for c in kgs:
            kg = kgs[c]
            
            if rel in kg and prompt['o'] in kg[rel]:
                results[c] = len(kg[rel])
                #print(c, kg[rel])
                
            if 'is' in kg and prompt['o'] in kg['is']:
                results[c] = len(kg[rel]) * 2
        
        if len(results) > 0:
            maxLen = max(list(results.values()))
            for c in results:
                results[c] = 1.0 - results[c] / maxLen
    
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))  
    print(results)
    return results


#instructions = "Generate the JSON knowledge graph in subject, relation, object format. Do not answer the question. Only include information from the prompt. All missing values must be set to \"Unknown\". The relation should be one of: have, color, predators, eats, found in, is, unknown"

#instructions = "Generate the JSON knowledge graph in subject, relation, object format. Do not answer the question. Only include information from the prompt. All missing values must be set to \"Unknown\"."

#kg_name_res("what are the predators of moon jellyfishes?", instructions)
#kg_name_res("find me images of the predators of moon jelly", instructions)
#kg_name_res("find me images of what moon jelly eat", instructions)
#kg_name_res("moon jellyfish", instructions)
#kg_name_res("creatures with tentacles", instructions)
#kg_name_res("find me images of creatures that are orange", instructions)
