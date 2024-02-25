import tkinter as tk
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

def check_pos(y, x, shape):
    return 0 <= x < shape[1] and 0 <= y < shape[0]

def similar_color_rgb(r1, g1, b1, r2, g2, b2, tolerance):
    if r2-tolerance <= r1 < r2+tolerance:
        if g2-tolerance <= g1 < g2+tolerance:
            if b2-tolerance <= b1 < b2+tolerance:
                return True
    return False

def similar_color_hsv(color, hc, sc, vc, rc, gc, bc):
    #white
    if color == 0:
        if rc > 170 and gc > 170 and bc > 170 and vc > 220 and hc > 60:
            return True
        else:
            return False
        
    elif color == 1:
        if 1 <  hc < 10 or 1 < vc < 75:
            return True
        else:
            return False 
        
    elif color == 2:
        if  160 < hc < 179 and vc > 75 and sc > 63:
            return True
        else:
            return False

    elif color == 3:  
        if 10 < hc < 25 and vc > 75 and sc > 63:
            return True
        else:
            return False
       
    elif color == 4:  
        if 25 < hc < 40 and vc > 75 and sc > 43:
            return True
        else:
            return False
 
    elif color == 5:  
        if 40 < hc < 75 and vc > 75 and sc > 63:
            return True
        else:
            return False  
        
    elif color == 6:  
        if 75 < hc < 110 and vc > 75 and sc > 83:
            return True
        else:
            return False  
       
    elif color == 7:  
        if 110 < hc < 130 and vc > 75 and sc > 83:
            return True
        else:
            return False
    
    elif color == 8:  
        if 130 < hc < 143 and vc > 75 and sc > 63:
            return True
        else:
            return False

    elif color == 9:  
        if 143 <  hc < 160 and vc > 75 and sc > 63:
            return True
        else:
            return False
 

def find_all_pattern(img_rgb, img_hsv, new_img, y, x):
    ro, go, bo = img_rgb[y, x]
    ho, so, vo = img_hsv[y, x]

    color = 0

    #white
    if ro > 170 and go > 170 and bo > 170 and vo > 220 and ho > 60:
        color = 0
    
    #black
    elif 1 < ho < 10 or 1 < vo < 75:
        color = 1

    #red
    elif 160 < ho < 179 and vo > 75 and so > 63:
        color = 2
    #orange
    elif 10 < ho < 25 and vo > 75 and so > 63:
        color = 3

    #yellow
    elif 25 < ho < 40 and vo > 75 and so > 43:
        color = 4

    #green
    elif 40 < ho < 75 and vo > 75 and so > 63:
        color = 5

    #brightBlue
    elif 75 < ho < 105 and vo > 75 and so > 83:
        color = 6
    
    #darkBlue
    elif 105 < ho < 130 and vo > 75 and so > 83:
        color = 7

    #purple
    elif 130 < ho < 143 and vo > 75 and so > 63:
        color = 8
    
    #pink
    elif 143 <  ho < 160 and vo > 75 and so > 63:
        color = 9

    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            rc, gc, bc = img_rgb[i, j]  
            hc, sc, vc = img_hsv[i, j]

            if similar_color_hsv(color, hc, sc, vc, rc, gc, bc):
                new_img[i, j] = img_rgb[i, j]
            else:
                new_img[i, j][0] = 100
                new_img[i, j][1] = 100
                new_img[i, j][2] = 100

def patternDivision(x, y, img):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    new_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i, j][0] = 100
            new_img[i, j][1] = 100
            new_img[i, j][2] = 100

    o_y = y
    o_x = x

    find_all_pattern(img_rgb, img_hsv, new_img, o_y, o_x)
    window_name = 'new_img'
    new_img = cv.cvtColor(new_img, cv.COLOR_RGB2BGR)
    return new_img

