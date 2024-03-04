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


def kg_name_res(prompt):
    with open('scripts/kg/kg_trees.json') as f:
        kgs = json.load(f)
        
    

    instructions = "Generate the JSON knowledge graph in subject, relation, object format. Do not answer the question. Only include information from the prompt. All missing values must be set to \"Unknown\"."

    answer = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-1106",
      timeout=30,
      messages=[
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt}
      ]
    )
    prompt_kg = json.loads(answer['choices'][0]['message']['content'])
    print(prompt_kg)
    
    if 'subject' in prompt_kg and 'relation' in prompt_kg and 'object' in prompt_kg:
        prompt_tree = {'nodes': [prompt_kg['subject'], prompt_kg['object']], 'edges': [{"p": prompt_kg['subject'], "c": prompt_kg['object'], "label": prompt_kg['relation']}]}
    else:
        return
    
    qnodes, qedges = getNodesEdges2(prompt_tree)
    prompt_embeddings = getEmbeddings(qnodes + qedges)
    prompt_embeddings['Unknown'] = 0

    results = []
    for c in kgs:
        if not os.path.isfile("scripts/kg/embeddings/"+c+".json"):
            continue
        
        try:        
            with open("scripts/kg/embeddings/"+c+".json") as f:
                embeddings = json.load(f)
        except:
            continue
            
        #if c != 'Aurelia aurita':
        #    continue
        
        pruned = prune(c, kgs[c], prompt_tree, embeddings, prompt_embeddings)
        if len(pruned) > 0:
            print(pruned)
            results.extend(pruned)
    
    return results


def isA(obj, cat, client, matches):
    obj = str(obj)
    cat = str(cat)
    
    if obj +' - '+ cat in matches:
        #print('Is "'+obj+'" a '+cat+'? '+str(matches[obj +' - '+ cat])+' (match found)')
        return matches[obj +' - '+ cat], matches

    if obj.lower().replace(' ', '') == cat.lower().replace(' ', ''):
        #print('Is "'+obj+'" a '+cat+'? Yes (exact match)')
        matches[obj +' - '+ cat] = True
        return True, matches

    instructions = 'Output "yes" or "no"'
    
    text = 'Is "'+obj+'" a '+cat+'?'

    try:
        answer = client.chat.completions.create(
          model="gpt-3.5-turbo-1106",
          timeout=10,
          messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text}
          ]
        )
    except:
        time.sleep(3)
        answer = client.chat.completions.create(
          model="gpt-3.5-turbo-1106",
          timeout=30,
          messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text}
          ]
        )
    
    res = answer.choices[0].message.content
    print(text +" - "+ res)
    
    match = res.lower().startswith('yes')
    matches[obj +' - '+ cat] = match
    return match, matches
    
    
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






def getName(kg):
    if isinstance(kg, dict):
        for k in kg:
            if not isinstance(kg[k], (dict, list)):
                return kg[k]
    else:
        return kg
    return ""

def getNodesEdges(kg, tree, root, edge=''):
    print('root = '+root)
    print(kg)
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
    print(concept)

    for kg in kgs:
        if isinstance(kg, str):
            print(kg)
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



def prune(concept, tree, tree2, embeddings, prompt_embeddings):
    #print(concept)
    
    aliases = [concept]
    for t in tree['edges']:
        #print(t)
        label = str(t['label']).lower()
        if 'name' in label or 'alias' in label:
            aliases.append(str(t['c']))
    #print(aliases)
    
    p = []
    for t in tree['edges']:
        s = str(t['p'])
        r = str(t['label'])
        o = str(t['c'])
        
        for q in tree2['edges']:
            qs = str(q['p'])
            qr = str(q['label'])
            qo = str(q['c'])

            #print(qs)
            
            if s not in embeddings or r not in embeddings or o not in embeddings:
                continue
            
            ss = np.dot(embeddings[s], prompt_embeddings[qs])
            for alias in aliases:
                if alias not in embeddings:
                    continue
                sa = np.dot(embeddings[alias], prompt_embeddings[qs])
                if sa > ss:
                    ss = sa
            
            sr = np.dot(embeddings[r], prompt_embeddings[qr])
            so = np.dot(embeddings[o], prompt_embeddings[qo])

            minscore = 0.8
            if (qs == 'Unknown' or ss > 0.95) and sr > minscore and (qo == 'Unknown' or so > minscore):
                #print(ss)
                p.append(t)
                #print(json.dumps(t) +" - " + json.dumps(q) + ' - '+str(ss)+" "+str(sr)+" "+str(so))
                break

    return p


    
#kg_name_res("what are the predators of moon jellyfishes?")
#kg_name_res("find me images of the predators of moon jellyfishes")
#kg_name_res("moon jellyfishes")
