import math
import json
import re
import random
from PIL import Image
import requests
import cv2
import numpy
import shutil


GOOD_BOUNDING_BOX_MIN_SIZE = 0.05
MIN_SHARPNESS = 20
INCLUDE_CUTOFF = False

f = open('data/good_images.json')
all_imgs = json.load(f)

for concept in all_imgs:
    imgs = all_imgs[concept]
    for img in imgs:
        coverage = (img['box']['w']*img['box']['h']) / (img['w']*img['h'])
        if (INCLUDE_CUTOFF or not img['cutoff']) and coverage > GOOD_BOUNDING_BOX_MIN_SIZE and img['sharpness'] > MIN_SHARPNESS:
            shutil.copyfile('data/imgs/'+img['filename'], 'data/imgs_filtered/'+img['filename'])
            