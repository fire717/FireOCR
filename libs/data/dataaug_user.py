 
from PIL import Image
import numpy as np
import pandas as pd
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
def addLine(img, p=0.5):


    return img


def randomPaste(img, p=0.5):
    colors = [(255,255,255), ()]

    return img

class TrainDataAug:
    def __init__(self, img_size):
        self.h = img_size[0]
        self.w = img_size[1]

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        #print(img.shape)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # raw_h, raw_w = img.shape[:2]
        # min_size = max(img.shape[:2])


        # img = A.ShiftScaleRotate(
        #                         shift_limit=0.1,
        #                         scale_limit=0.1,
        #                         rotate_limit=3,
        #                         interpolation=cv2.INTER_LINEAR,
        #                         border_mode=cv2.BORDER_CONSTANT,
        #                          value=0, mask_value=0,
        #                         p=1)(image=img)['image']

        # img = A.GridDistortion(num_steps=5, distort_limit=0.2,
        #                     interpolation=1, border_mode=4, p=1)(image=img)['image']
        #GridDropout
        # img = A.OneOf([A.ShiftScaleRotate(
        #                         shift_limit=0.1,
        #                         scale_limit=0.1,
        #                         rotate_limit=30,
        #                         interpolation=cv2.INTER_LINEAR,
        #                         border_mode=cv2.BORDER_CONSTANT,
        #                          value=0, mask_value=0,
        #                         p=0.5),
        #                 A.GridDistortion(num_steps=5, distort_limit=0.2,
        #                     interpolation=1, border_mode=4, p=0.4)],
        #                 p=0.5)(image=img)['image']
        
        # img = A.HueSaturationValue(hue_shift_limit=4, 
        #                 sat_shift_limit=4, val_shift_limit=4,  p=1)(image=img)['image']

        # img = A.OneOf([A.RandomBrightness(limit=0.1, p=1), 
        #             A.RandomContrast(limit=0.1, p=1),
        #             A.RandomGamma(gamma_limit=(50, 100),p=1),
        #             A.HueSaturationValue(hue_shift_limit=4, 
        #                 sat_shift_limit=4, val_shift_limit=4,  p=1)], 
        #             p=0.6)(image=img)['image']

        
        # img = A.Resize(self.h,self.w,cv2.INTER_LANCZOS4,p=1)(image=img)['image']
        # img = A.OneOf([A.GaussianBlur(blur_limit=(3,7), p=0.1),
        #                 A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
        #                 A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.4)], 
        #                 p=0.4)(image=img)['image']


        
        img = Image.fromarray(img)
        
        return img


class TestDataAug:
    def __init__(self, img_size, resize_h=32):
        self.h = img_size[0]
        self.w = img_size[1]
        self.resize_h = resize_h

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


        h,w = img.shape[:2]
        resize_w = int(w*self.resize_h/h)+1
        img = A.Resize(self.resize_h,resize_w,cv2.INTER_LANCZOS4,p=1)(image=img)['image']
        img = Image.fromarray(img)
        return img



