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
MIN_SHARPNESS = 20
TOPN = 20

def getBorder(img):
    image = numpy.array(img)
    y_nonzero, x_nonzero, _ = numpy.nonzero(image)
    return numpy.min(y_nonzero), numpy.max(y_nonzero), numpy.min(x_nonzero), numpy.max(x_nonzero)

def marginGood(margin_width, image_width):
    return margin_width / image_width > GOOD_BOUNDING_BOX_MIN_MARGINS
    
def getImageProperties(d, b):
    img = Image.open(requests.get(d['url'], stream=True).raw)
    img = img.convert('RGB')
    
    marginx = GOOD_BOUNDING_BOX_MIN_MARGINS * d['width']
    marginy = GOOD_BOUNDING_BOX_MIN_MARGINS * d['height']
    
    ytop, ybtm, xleft, xright = getBorder(img)
    allMarginsGood = marginGood(b['x']-xleft, xright-xleft) and marginGood(xright-(b['x']+b['width']), xright-xleft) \
      and marginGood(b['y']-ytop, ybtm-ytop) and marginGood(ybtm-(b['y']+b['height']), ybtm-ytop)
    
    fname = b['concept'].replace('"', '').replace('/', '').replace('.', '').replace(' ','_')+'_'+d['uuid']+'.jpg'
    #img_saved = img.crop((xleft, ytop, xright, ybtm))
    #img_saved.save('data/imgs/'+fname)
    
    
    if b['y']+marginy < b['y']+b['height']-marginy:
        box = img.crop((max(xleft, b['x']-marginx*2), b['y']+marginy, min(xright, b['x']+b['width']+marginx*2), b['y']+b['height']-marginy))
    else:
        box = img.crop((max(xleft, b['x']-marginx*2), b['y'], min(xright, b['x']+b['width']+marginx*2), b['y']+b['height']))
    box_save = img.crop((
        max(xleft, b['x']-marginx), 
        max(ytop, b['y']-marginy), 
        min(xright, b['x']+b['width']+marginx), 
        min(ybtm, b['y']+b['height']+marginy),
    ))
    box_save.save('data/imgs/'+fname)

    img_grey = cv2.cvtColor(numpy.array(box), cv2.COLOR_BGR2GRAY)
    laplacian_image = cv2.Laplacian(img_grey, cv2.CV_64F)
    variance = numpy.var(laplacian_image)
    
    return variance, fname, not allMarginsGood


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
        sharpness, fname, cutoff = getImageProperties(d, box)
        scores[d['uuid']]['sharpness'] = sharpness
        scores[d['uuid']]['fname'] = fname
        scores[d['uuid']]['cutoff'] = cutoff
        
    
    return data, scores
  

concepts = boundingboxes.find_concepts()[1:]
#concepts = ['Aurelia aurita']

f = open('data/good_images.json')
good_imgs = {} 
good_imgs = json.load(f)

f = open('data/good_images_errors.json')
errors = json.load(f)

for concept in concepts:
    if concept in good_imgs:
        continue
        
    print(concept)
    try:
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
            img['sharpness'] = scores[d['uuid']]['sharpness']
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
    except:
        errors.append(concept)
        with open("data/good_images_errors.json", "w") as outfile:
            json.dump(errors, outfile)