import math
import json
import re
import random
from PIL import Image
import requests
import cv2
import numpy

from fathomnet.api import regions
from fathomnet.api import taxa
from fathomnet.api import boundingboxes
from fathomnet.api import images
from fathomnet.dto import GeoImageConstraints


GOOD_BOUNDING_BOX_MIN_SIZE = 0 #0.05
GOOD_BOUNDING_BOX_MIN_MARGINS = 0.01
MIN_BLUR = 50
TOPN = 20

def getBorder(img):
    image = numpy.array(img)
    y_nonzero, x_nonzero, _ = numpy.nonzero(image)
    return numpy.min(y_nonzero), numpy.max(y_nonzero), numpy.min(x_nonzero), numpy.max(x_nonzero)

def marginGood(margin_width, image_width):
    return margin_width / image_width > GOOD_BOUNDING_BOX_MIN_MARGINS
    
def getBlurriness(d, b):
    img = Image.open(requests.get(d['url'], stream=True).raw)
    img = img.convert('RGB')
    
    marginx = 0.01*d['width']
    marginy = 0.01*d['height']
    
    ytop, ybtm, xleft, xright = getBorder(img)
    allMargingGood = marginGood(b['x']-xleft, xright-xleft) and marginGood(xright-(b['x']+b['width']), xright-xleft) \
      and marginGood(b['y']-ytop, ybtm-ytop) and marginGood(ybtm-(b['y']+b['height']), ybtm-ytop)
    
    fname = b['concept'].replace('"', '').replace('/', '').replace('.', '').replace(' ','_')+'_'+d['uuid']
    img_saved = img.crop((xleft, ytop, xright, ybtm))
    img_saved.save('data/imgs/'+fname+'.jpg')
    
    
    
    if d['width'] - b['width'] < marginx*2 and d['height'] - b['height'] < marginy*2:
        img_grey = cv2.cvtColor(numpy.array(img), cv2.COLOR_BGR2GRAY)
        laplacian_image = cv2.Laplacian(img_grey, cv2.CV_64F)
        img.save('data/imgs/'+fname+'-box.jpg')
        return numpy.var(laplacian_image), fname, not allMargingGood
    
    box = img.crop((b['x'], b['y'], b['x']+b['width'], b['y']+b['height']))
    
    box_saved = img.crop((
        max(xleft, b['x']-marginx), 
        max(ytop, b['y']-marginy), 
        min(xright, b['x']+b['width']+marginx), 
        min(ybtm, b['y']+b['height']+marginy),
    ))
    box_saved.save('data/imgs/'+fname+'-box.jpg')
    
    side = img.crop((b['x']-marginx*2, b['y'], b['x']-marginx, b['y']+b['height']))
            
    #side.save('data/imgs/'+fname+'-side.jpg')
    
    box = cv2.cvtColor(numpy.array(box), cv2.COLOR_BGR2GRAY)
    side = cv2.cvtColor(numpy.array(side), cv2.COLOR_BGR2GRAY)
    
    return abs(box.mean() - side.mean()), fname, not allMargingGood


def boundingBoxQualityScore(d, names):
  # the score is the max size of the bounding boxes divided by the image size
  
  score = 0
  uuid = 0
  count = 0
  if d['width']*d['height'] == 0:
    return 0, 0
  for box in d['boundingBoxes']:
    if box['concept'] not in names:
      continue
    count = count + 1
    

    s = (box['width']*box['height'])
    if s > score:
      score = s
      uuid = box['uuid']
  if count == 0:
    return 0, 0
  return score/(d['width']*d['height']), uuid

def filterByBoundingBoxes(data, names):


    scores = {}
    for d in data:
        s, uuid = boundingBoxQualityScore(d, names)
        scores[d['uuid']] = {'score': s, 'box_id': uuid}
    data = [d for d in data if scores[d['uuid']]['score'] > GOOD_BOUNDING_BOX_MIN_SIZE]

    data.sort(key=lambda d: scores[d['uuid']]['score'], reverse=True)
    data = data[:TOPN]
    
    for d in data:
        box = {}
        for b in d['boundingBoxes']:
            if b['uuid'] == scores[d['uuid']]['box_id']:
                box = b
                break
        blurriness, fname, cutoff = getBlurriness(d, box)
        scores[d['uuid']]['blurriness'] = blurriness
        scores[d['uuid']]['fname'] = fname
        scores[d['uuid']]['cutoff'] = cutoff
        

    #data = [d for d in data if blurriness[d['uuid']] > MIN_BLUR]
    
    return data, scores
  

concepts = boundingboxes.find_concepts()[1:]
#concepts = ['Aurelia aurita']

f = open('data/good_images.json')
good_imgs = {} 
good_imgs = json.load(f)
for concept in concepts:
    if concept in good_imgs:
        continue
        
    print(concept)
    constraints = GeoImageConstraints(
        concept=concept, 
      )
    data = images.find(constraints)
    data = [d.to_dict() for d in data]
    data, scores = filterByBoundingBoxes(data, [concept])
    imgs = []
    for d in data:
        img = {}
        img['id'] = d['uuid']
        img['url'] = d['url']
        img['w'] = d['width']
        img['h'] = d['height']
        img['score'] = scores[d['uuid']]['score']
        img['cutoff'] = scores[d['uuid']]['cutoff']
        img['blurriness'] = scores[d['uuid']]['blurriness']
        img['filename'] = scores[d['uuid']]['fname']
        for b in d['boundingBoxes']:
            if b['uuid'] == scores[d['uuid']]['box_id']:
                img['box'] = {'x': b['x'], 'y': b['y'], 'w': b['width'], 'h': b['height']}
                break
        if 'box' in img:
            imgs.append(img)
        
    good_imgs[concept] = imgs

    with open("data/good_images.json", "w") as outfile:
        json.dump(good_imgs, outfile)
