import tkinter as tk
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import time

def check_pos(y, x, shape):
    return 0 <= x < shape[1] and 0 <= y < shape[0]

def similar_color_rgb(r1, g1, b1, r2, g2, b2, tolerance):
    if r2-tolerance <= r1 < r2+tolerance:
        if g2-tolerance <= g1 < g2+tolerance:
            if b2-tolerance <= b1 < b2+tolerance:
                return True
    return False

def similar_color_hsv(hc, sc, vc, ho, so, vo, rc, gc, bc, ro, go, bo):
    if ro > 170 and go > 170 and bo > 170 and ho > 95:
        if rc > 170 and gc > 170 and bc > 170 and hc > 95:
            return True
        else:
            return False
    else:
        #red
        if ho < 10 and ho > 340:
            if hc < 10 and hc > 340 and vc > 60 and sc > 63:
                return True

        #orange 
        elif 10 < ho < 40:
            if 10 < hc < 40 and vc > 60 and sc > 63:
                return True
            
        #yellow 
        elif 40 < ho < 80:
            if 40 < hc < 80 and vc > 60 and sc > 43:
                return True
        
        #green 
        elif 80 < ho < 150:
            if  80 < hc < 150 and vc > 60 and sc > 63:
                return True
    
        #blue 
        elif 150 < ho < 270:
            if 150 < hc < 270 and vc > 60 and sc > 83:
                return True  

        #purple 
        elif 270 < ho < 310:
            if 270 < hc < 310 and vc > 60 and sc > 63:
                return True

        #pink 
        elif 310 < ho < 340:
            if 310<  hc < 340 and vc > 60 and sc > 63:
                return True
    return False
    
def find_all_pattern(img_rgb, img_hsv, new_img, y, x):
    
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            cur_r, cur_g, cur_b = img_rgb[i, j]  
            ori_r, ori_g, ori_b = img_rgb[y, x]

            cur_h, cur_s, cur_v = img_hsv[i, j]
            ori_h, ori_s, ori_v = img_hsv[y, x]
            
            #if similar_color_rgb(cur_r, cur_g, cur_b, ori_r, ori_g, ori_b, 50):
            #    new_img[i, j] = img_rgb[i, j]

            if similar_color_hsv(cur_h, cur_s, cur_v, ori_h, ori_s, ori_v, cur_r, cur_g, cur_b, ori_r, ori_g, ori_b):
                new_img[i, j] = img_rgb[i, j]

def patternDivision(x, y, img):
    start_time = time.time()
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    new_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    vec = [[-1, -1], [-1, 0], [-1, 1],
           [0, -1], [0, 0], [0, 1],
           [1, -1], [1, 0], [1, 1]]
    o_y = y
    o_x = x

    find_all_pattern(img_rgb, img_hsv, new_img, o_y, o_x)
    window_name = 'new_img'
    new_img = cv.cvtColor(new_img, cv.COLOR_RGB2BGR)

    time_taken = time.time() - start_time

    print(f"INteract Time taken: {time_taken} seconds")
    return new_img

