import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
import torch

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([0, 0, 0, 1.0])
    h, w = mask.shape[-2:]
    mask_image = (mask.reshape(h, w, 1)) * color.reshape(1, 1, -1)
    return mask_image

def segment(x, y, image):
    # Get the mouse position
    PointX, PointY = y, x
    from segment_anything import sam_model_registry, SamPredictor

    #print(PointX, PointY)
    #print(os.getcwd())


    sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    
    predictor.set_image(image)
    
    input_point = np.array([[PointY, PointX]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    masked_image = []
    for i in range(0,3):
        mask = show_mask(masks[i], plt.gca())
        mask = np.dstack((mask[:,:,3], np.dstack((mask[:,:,3], mask[:,:,3]))))
        masked_image.append((image*mask).astype(int))
        #print(image*mask)
    
    return masked_image