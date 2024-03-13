import cv2
import numpy as np
from matplotlib import pyplot as plt

def cropping(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    non_black_coords = np.argwhere(gray != 0)

    min_y, min_x = non_black_coords.min(axis=0)
    max_y, max_x = non_black_coords.max(axis=0)

    cropped = np.zeros_like(img)
    cropped = img[min_y-10:max_y+10, min_x-10:max_x+10]
    cropped = cv2.resize(cropped, dsize=(800, 600), interpolation=cv2.INTER_CUBIC)
    return cropped
