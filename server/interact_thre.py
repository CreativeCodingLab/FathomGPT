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

def similar_color_hsv(color, hc, sc, vc, rc, gc, bc):
    #white
    if color == 0:
        if (rc > 165 and gc > 165 and bc > 165) or (75 <= hc < 105 and vc > 180 and sc > 180):
            return True
        
    elif color == 1:
        if  1 < vc < 65:
            return True 
        
    elif color == 2:
        if  (0 <= hc < 10 and sc > 120 and vc > 75) or (160 < hc < 179 and vc > 75 and sc > 63):
            return True

    elif color == 3:  
        if 10 <= hc < 25 and vc > 75 and sc > 63:
            return True
       
    elif color == 4:  
        if 25 <= hc < 40 and vc > 75 and sc > 43:
            return True
 
    elif color == 5:  
        if 40 <= hc < 75 and vc > 75 and sc > 63:
            return True  
        
    elif color == 6:  
        if 75 <= hc < 110 and vc > 75 and sc > 83:
            return True  
       
    elif color == 7:  
        if 110 <= hc < 130 and vc > 75 and sc > 40:
            return True
    
    elif color == 8:  
        if 130 <= hc < 143 and vc > 75 and sc > 63:
            return True

    elif color == 9:  
        if 143 <=  hc < 160 and vc > 75 and sc > 63:
            return True
        
    return False

def find_all_pattern(color_thre, img_rgb, img_hsv, new_img, y, x):
    ro, go, bo = img_rgb[y, x]
    ho, so, vo = img_hsv[y, x]

    color = 0

    #white
    if (ro > 165 and go > 165 and bo > 165) or (75 <= ho < 105 and vo > 180 and so > 180):
        color = 0
    
    #black
    elif 1 < vo < 65:
        color = 1

    #red
    elif (1 <= ho < 10 and so > 120 and vo > 75) or (160 < ho < 179 and vo > 75 and so > 63):
        color = 2

    #orange
    elif 10 <= ho < 25 and vo > 75 and so > 63:
        color = 3

    #yellow
    elif 25 <= ho < 40 and vo > 75 and so > 43:
        color = 4

    #green
    elif 40 <= ho < 75 and vo > 75 and so > 63:
        color = 5

    #brightBlue
    elif 75 <= ho < 105 and vo > 75 and so > 83:
        color = 6
    
    #darkBlue
    elif 105 <= ho < 130 and vo > 75 and so > 40:
        color = 7

    #purple
    elif 130 <= ho < 143 and vo > 75 and so > 63:
        color = 8
    
    #pink
    elif 143 <= ho < 160 and vo > 75 and so > 63:
        color = 9

    Color_thre_table = [[0, 6, 7], [1, 1, 1], [2, 3, 9], [3, 2, 4], [4, 3, 5], [5, 4, 6], [6, 7, 0], [7, 6, 0], [8, 7, 9], [9, 2, 8]]
    

    cal_time = -1.0
    start = time.time()
    print("img_rgb.shape: ", img_rgb.shape[0], img_rgb.shape[1])
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            rc, gc, bc = img_rgb[i, j]  
            hc, sc, vc = img_hsv[i, j]
            start1 = time.time()
            for k in range(0, color_thre):
                if similar_color_hsv(Color_thre_table[color][k], hc, sc, vc, rc, gc, bc):
                    new_img[i, j] = img_rgb[i, j]
            cal_time = max(cal_time, time.time()-start1)

    print("huge ugly loop: ", time.time()-start, "cal_time: ", cal_time)

# division
# @param color_thre : the number of color threshold (1,2,3)
def patternDivision(x, y, img, color_thre):
    
    start = time.time()
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    new_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    print("fist time: ", time.time()-start)
    start = time.time()

    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i, j][0] = 100
            new_img[i, j][1] = 100
            new_img[i, j][2] = 100

    o_y = y
    o_x = x

    print("second time: ", time.time()-start)
    start = time.time()

    find_all_pattern(color_thre, img_rgb, img_hsv, new_img, o_y, o_x)


    print("third time: ", time.time()-start)
    start = time.time()

    window_name = 'new_img'
    new_img = cv.cvtColor(new_img, cv.COLOR_RGB2BGR)

    print("fourth time: ", time.time()-start)
    return new_img