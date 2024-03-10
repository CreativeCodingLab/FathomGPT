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

def matchColor(dh, uh, ds, us, dv, uv, img):
    mask_h = (img[:,:,0] <= uh) & (img[:,:,0] >= dh)
    mask_s = (img[:,:,1] <= us) & (img[:,:,1] >= ds)
    mask_v = (img[:,:,2] <= uv) & (img[:,:,2] >= dv)

    combined_mask = mask_h & mask_s & mask_v
    result = np.zeros_like(img)
    result[combined_mask] = img[combined_mask]
    return result

def mergeTwoImage(img1, img2):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    mask = gray1 > gray2
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    merged_img = np.where(mask_3d, img1, img2)
    return merged_img

def findColor(color_thre, img_rgb, img_hsv, y, x):
    ro, go, bo = img_rgb[y, x]
    ho, so, vo = img_hsv[y, x]
    color = 0

    colorRange = [[0,0,0], [1,3,9], [2,3,9], [3, 1, 4], [4, 3, 5], [5, 4, 6], [6, 0, 7], [7, 6, 8], [8, 7, 9], [9, 1, 3], [10, 10, 10]]
   
    #dh, uh, ds, us, dv, uv
    colorTable = [[0, 179, 0, 75, 210, 255],    #white      0
                  [0, 10, 100, 255, 50, 255],       #red        1
                  [160, 179, 100, 255, 50, 255],    #red        2
                  [10, 25, 75, 255, 50, 255],       #orange     3
                  [25, 40, 75, 255, 50, 255],       #yellow     4
                  [40, 75, 75, 255, 50, 255],       #green      5
                  [75, 110, 75, 255, 50, 255],      #brightBlue 6
                  [110, 130, 75, 255, 50, 255],     #darkBlue   7 
                  [130, 143, 75, 255, 50, 255],     #purple     8
                  [143, 160, 75, 255, 50, 255],     #pink       9
                  [0, 179, 0, 255, 1, 50]]          #black      10
    
    for i in range(11):  
        if (colorTable[i][0] < ho <= colorTable[i][1]) and (colorTable[i][2] < so <= colorTable[i][3]) and (colorTable[i][4] < vo <= colorTable[i][5]):
            color = i
            break

    new_img = np.zeros_like(img_hsv)

    for k in range(0, color_thre):
        
        color_n = colorRange[color][k]
        if (color_n != 1 and color_n != 2):
            dh, uh, ds, us, dv, uv = colorTable[color_n]
            new_img = mergeTwoImage(new_img, matchColor( dh, uh, ds, us, dv, uv, img_hsv))
        else:
            for color_n in range(1,3):
                dh, uh, ds, us, dv, uv = colorTable[color_n]
                new_img = mergeTwoImage(new_img, matchColor( dh, uh, ds, us, dv, uv, img_hsv))

    return new_img

def patternDivision(x, y, img, color_thre):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    result = np.zeros_like(img_hsv)
    result = findColor(color_thre, img_rgb, img_hsv, y, x)
    result = cv.cvtColor(result, cv.COLOR_HSV2BGR)

    is_black = np.all(result == [0, 0, 0], axis=-1)
    new_background_color = [100, 100, 100]
    result[is_black] = new_background_color
    return result