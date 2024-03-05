import openai
import json
from datetime import datetime
import math
from urllib.request import urlopen
from urllib.parse import quote

# load and set our key
openai.api_key = "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"

from fathomnet.api import images
from fathomnet.api import regions
from fathomnet.api import taxa
from fathomnet.api import boundingboxes
from fathomnet.dto import GeoImageConstraints

from dotenv import load_dotenv
import os
from nltk.stem import PorterStemmer


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")


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
        answer = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-0125",
          timeout=30,
          messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text}
          ]
        )
        answer = json.loads(answer['choices'][0]['message']['content'].replace('```json\n', '').replace('```', ''))
    
    return answer


def embed(terms):
    terms = [str(t) for t in terms if str(t) != '']
    
    resp = openai.Embedding.create(
        input=terms,
        engine="text-embedding-ada-002")

    embeddings = [r['embedding'] for r in resp['data']]
    return dict(zip(terms, embeddings))
    

relations = {
    'other names': 'alias', 
    'colors': 'colors', 
    'body parts': 'has', 
    'predators': 'predators', 
    'diet': 'eats', 
    'locations': 'found in', 
    'environments': 'found in', 
    'characteristics': 'is',
}


def genKg():
    with open('scripts/kg/descriptions.json') as f:
        data = json.load(f)

    with open('scripts/kg/kg.json') as f:
        kg = json.load(f)

    for concept in data: 
        text = data[concept]['wiki']
        
        if concept in kg:
            continue
        
        print(concept)
        kg[concept] = {}
        
        for r in relations:
            instructions = "Given the information, what are the "+r+" of "+concept+"? The output must be a machine-readable JSON list of terms and nothing else."

            features = None
            for i in range(5):
                features = callGPT(instructions, text)
                #print(features)
                if isinstance(features, list) and len(features) > 0:
                    break
            if not features:
                continue
            
            k = relations[r]
            if k not in kg[concept]:
                kg[concept][k] = []
            features = [f.lower() for f in features]
            kg[concept][k].extend(list(set(features)))
            
        with open("scripts/kg/kg.json", "w") as outfile:
            json.dump(kg, outfile)


def genEmbeddings():
    em = embed(relations.values())
    with open("scripts/kg/relations.json", "w") as outfile:
        json.dump(em, outfile)


def genNormalized():
    with open('scripts/kg/kg.json') as f:
        kgs = json.load(f)
        
    ps = PorterStemmer()

    for c in kgs:
        kg = kgs[c]
        for rel in kg:
            kgs[c][rel] = [ps.stem(t) for t in kg[rel]]
    
    with open("scripts/kg/kg_stemmed.json", "w") as outfile:
        json.dump(kgs, outfile)


#genKg()
genEmbeddings()
#genNormalized()
