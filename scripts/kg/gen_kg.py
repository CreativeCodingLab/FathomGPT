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

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")


def genTrees():
    with open('scripts/kg/kg_raw.json') as f:
        kgs = json.load(f)
        
    with open('scripts/kg/kg_trees.json') as f:
        trees = json.load(f)
        
    for c in kgs:
        if c in trees:
            continue
            
        print(c)
        
        kg = kgs[c]
        tree = {'nodes': [], 'edges': []}
        tree = getTree([kg], tree, c)
        trees[c] = tree
        
        with open("scripts/kg/kg_trees.json", "w") as outfile:
            json.dump(trees, outfile)


def genKgs():
    instructions = "Generate the JSON knowledge graph"

    with open('scripts/kg/descriptions.json') as f:
        data = json.load(f)

    with open('scripts/kg/kg_raw.json') as f:
        kgs = json.load(f)

    i = 0
    for concept in data:
        if concept in kgs:
            continue
            
        print(concept)
        
        text = data[concept]['wiki']
        
        try:
            answer = openai.ChatCompletion.create(
              model="gpt-3.5-turbo-1106",
              timeout=30,
              messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": text}
              ]
            )
            #print(answer)
        except:
            answer = openai.ChatCompletion.create(
              model="gpt-3.5-turbo-1106",
              timeout=30,
              messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": text}
              ]
            )
            
        try:
            kg = json.loads(answer['choices'][0]['message']['content'])
        except:
            print('---- err')
            continue
        
        kgs[concept] = json.dumps(kg)
        
        if i%10 == 0:
            with open("scripts/kg/kg_raw.json", "w") as outfile:
                json.dump(kgs, outfile)
        
        i = i+1
        #break
        
    with open("scripts/kg/kg_raw.json", "w") as outfile:
        json.dump(kgs, outfile)
    




def getName(kg):
    if isinstance(kg, dict):
        for k in kg:
            if not isinstance(kg[k], (dict, list)):
                return kg[k]
    else:
        return kg
    return ""

def getNodesEdges(kg, tree, root, edge=''):
    #print('root = '+root)
    #print(kg)
    if root not in tree['nodes']:
        tree['nodes'].append(root)
    if isinstance(kg, dict):
        for edge in kg:
            child = kg[edge]
            if isinstance(child, dict):
                tree['edges'].append({'p': root, 'c': edge, 'label': ''})
                tree = getNodesEdges(child, tree, edge)
                #for s in child:
                    #tree['edges'].append({'p': root, 'c': s, 'label': edge})
                    #tree = getNodesEdges(child[s], tree, s)
                    #tree = getNodesEdges(child[s], tree, edge, s)
            else:
                tree = getNodesEdges(child, tree, root, edge)
    elif isinstance(kg, list):
        #tree['edges'].append({'p': root, 'c': edge, 'label': ''})
        for c in kg:
            tree['edges'].append({'p': root, 'c': getName(c), 'label': edge})
            tree = getNodesEdges(c, tree, getName(c), edge)
            #tree['edges'].append({'p': root, 'c': c, 'label': edge})
    else:
        if str(kg).lower() in ['he', 'she', 'it', 'they', 'this', 'that']:
            return tree
        if root == kg:
            return tree
        tree['edges'].append({'p': root, 'c': kg, 'label': edge})
    return tree


def getTree(kgs, tree, concept):
    #print(concept)

    for kg in kgs:
        if isinstance(kg, str):
            #print(kg)
            kg = json.loads(kg)
        if concept:
            for root in kg:
                #tree['edges'].append({'p': concept, 'c': root, 'label': root})
                tree = getNodesEdges(kg[root], tree, concept, root)
            """
            if len(kg) == 1:
                for root in kg:
                    tree = getNodesEdges(kg[root], tree, concept)
            else:
                tree = getNodesEdges(kg, tree, concept)
            """
        else:
            for root in kg:
                tree = getNodesEdges(kg[root], tree, root)
    return tree


def getEmbeddings(terms):
    terms = [str(t) for t in terms if str(t) != '']
    
    resp = openai.Embedding.create(
        input=terms,
        engine="text-embedding-ada-002")

    embeddings = [r['embedding'] for r in resp['data']]
    return dict(zip(terms, embeddings))
    

def getNodesEdges2(kg):
    nodes = kg['nodes']

    edges = [t['label'] for t in kg['edges']]
    return nodes, edges


def genEmbed():

    with open('scripts/kg/kg_trees.json') as f:
        trees = json.load(f)
    
    
        
    for c in trees:    

        print(c)
        
        nodes, edges = getNodesEdges2(trees[c])

        embeddings = getEmbeddings(nodes + edges)
        
        with open("scripts/kg/embeddings/"+c+".json", "w") as outfile:
            json.dump(embeddings, outfile)


#genKgs()
#genTrees()
genEmbed()
