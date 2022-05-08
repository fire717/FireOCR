 
from PIL import Image
import numpy as np
# import pandas as pd
import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import cv2
import albumentations as A
import json
import platform




###### 1.Data aug
def addLine(img, p=0.2):
    if random.random()<p:
        h,w = img.shape[:2]

        color = random.randint(0,70)
        # thickness = random.randint(1,1)

        y = random.randint(1,h-2)
        #print(y)
        cv2.line(img, (0,y), (w-1,y), (color,color,color), 1)
    return img


def randomPaste(img, p=1):
    if random.random()<p:
        h,w = img.shape[:2]

        count = random.randint(1,6)
        count = min(w//h, count)

        
        for _ in range(count):

            x0 = random.randint(0,w-10)
            y0 = random.randint(0,h-10)
            x1 = min(w-1, x0+8)
            y1 = min(h-1, y0+8)

            # if random.random()<0.4:
            #     #black
            #     color = random.randint(0,70)
            #     colors = (color,color,color)
            # else:
            #     #red
            color = random.randint(180,230)
            colors = (color,random.randint(0,20),random.randint(0,20))

            cv2.rectangle(img, (x0,y0), (x1,y1), colors, random.randint(1,6))


    return img


def randomResize(img, ratio=0.6):
    if random.random()<ratio:
        resize_ratio = random.random()*0.5+0.3
        
        h,w = img.shape[:2]
        if h<30:
            return img
        resize_h = int(h*resize_ratio)
        resize_w = int(w*resize_ratio)

        img = cv2.resize(img, (resize_w, resize_h))
        #random.choice(resize_type)
        img = cv2.resize(img, (w,h))

    return img


class TrainDataAug:
    def __init__(self, img_size):
        self.h = img_size[0]
        self.w = img_size[1]

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        #print(img.shape)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # raw_h, raw_w = img.shape[:2]
        # min_size = max(img.shape[:2])


        # img = A.RandomBrightnessContrast(brightness_limit=0.1, 
        #                             contrast_limit=0.1, p=0.5)(image=img)['image']
        img = randomResize(img)

        img = A.GaussNoise(var_limit=(2.0, 5.0), mean=0, p=0.5)(image=img)['image']

        img = A.RGBShift(r_shift_limit=50,
                            g_shift_limit=50,
                            b_shift_limit=50,
                            p=0.3)(image=img)['image']

        
        img = A.ShiftScaleRotate(
                                    shift_limit=0.02,
                                    scale_limit=0.04,
                                    rotate_limit=0,
                                    interpolation=cv2.INTER_LINEAR,
                                    border_mode=cv2.BORDER_CONSTANT,
                                     value=(0,0,0), mask_value=0,
                                    p=0.5)(image=img)['image']

        # img = A.GridDistortion(num_steps=5, distort_limit=0.2,
        #                     interpolation=1, border_mode=0, 
        #                     p=0.3)(image=img)['image']


        img = addLine(img, p=0.2)
        # img = randomPaste(img, p=0.2)

        if len(img.shape)==3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        if random.random()<0.5:
            img = 255-img

        img = Image.fromarray(img)
        
        return img


class TestDataAug:
    def __init__(self, img_size, resize_h=40):
        self.h = img_size[0]
        self.w = img_size[1]
        self.resize_h = resize_h

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if len(img.shape)==3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # print("TestDataAug 1: ",img.shape)
        # img = randomResize(img)


        

        # print("TestDataAug 2: ",img.shape)
        img = Image.fromarray(img)

        # img = np.transpose(img,(2,0,1))
        # img = np.array(img, dtype=np.float32)
        # print(img[:,0])

        # img /= 255.
        # img = (img-0.5)
        # print(img[:,0])
        # img = np.reshape(img, (1, img.shape[0], img.shape[1]))
        # img = torch.from_numpy(img)
        # print(img.shape)
        # b
        return img



