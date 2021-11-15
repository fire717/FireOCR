 
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


from libs.data.dataaug_user import TrainDataAug, TestDataAug


##### Common
def getFileNames(file_dir, tail_list=['.png','.jpg','.JPG','.PNG']): 
        L=[] 
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] in tail_list:
                    L.append(os.path.join(root, file))
        return L




######## dataloader



class TensorDatasetTrainClassify(Dataset):
    _print_times = 0
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform

        # self.label_dict = {}
        # self.getLabels()
        # self.cate_dirs = []

    # def getLabels(self, img_path):
    #     label_path = os.path.join(self.label_dir, os.path.basename(img_path)[:-3]+'txt')

    #     with open(label_path, 'r') as f:
    #         line = f.readlines()[0]
    #     label = np.array([int(x) for x in line.strip().split(',')])
    #     return label


    def __getitem__(self, index):

        items = self.data[index].strip().split(" ")
        img_name = items[0]
        label = np.array([int(x) for x in items[1:]])

        # print(self.img_dir, img_name)
        img = cv2.imread(os.path.join(self.img_dir, img_name))

        if self.transform is not None:
            img = self.transform(img)

        # y = self.getLabels(self.data[index])

        # y_onehot = [0,0]
        # y_onehot[y] = 1

        return img, label, self.data[index]
        
    def __len__(self):
        return len(self.data)


class TensorDatasetTestClassify(Dataset):

    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        # img = cv2.imread(self.train_jpg[index])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img,(180, 180))

        #img = Image.open(self.train_jpg[index]).convert('RGB')
        img = cv2.imread(self.train_jpg[index])
        #img = imgPaddingWrap(img)
        #b
        if self.transform is not None:
            img = self.transform(img)

        # path_dir = '/'.join(self.train_jpg[index].split('/')[:-1])
        # y = 0
        # if  'true' in path_dir:
        #     y = 1

        return img, self.train_jpg[index]

    def __len__(self):
        return len(self.train_jpg)


###### 3. get data loader 




def getDataLoader(mode, input_data, cfg):




    data_aug_train = TrainDataAug(cfg['img_size'])
    data_aug_test = TestDataAug(cfg['img_size'])




    if mode=="test":
        my_dataloader = TensorDatasetTestClassify

        test_loader = torch.utils.data.DataLoader(
                my_dataloader(input_data[0],
                                transforms.Compose([
                                    data_aug_test,
                                    transforms.ToTensor(),
                                    my_normalize
                                ])
                ), batch_size=cfg['test_batch_size'], shuffle=False, 
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
            )

        return test_loader


    elif mode=="trainval":
        my_dataloader = TensorDatasetTrainClassify
        

        img_dir = cfg['img_dir']

        train_loader = torch.utils.data.DataLoader(
                                        my_dataloader(input_data[0],
                                            img_dir,
                                            transforms.Compose([
                                                data_aug_train,
                                                transforms.ToTensor(),
                                        ])),
                                batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

        val_loader = torch.utils.data.DataLoader(
                                        my_dataloader(input_data[1],
                                            img_dir,
                                            transforms.Compose([
                                                data_aug_test,
                                                transforms.ToTensor(),
                                        ])),
                                batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])
        return train_loader, val_loader


    elif mode=="train":
        my_dataloader = TensorDatasetTrainClassify
        
        #auto aug

        #from .autoaugment import ImageNetPolicy
        # from libs.FastAutoAugment.data import  Augmentation
        # from libs.FastAutoAugment.archive import fa_resnet50_rimagenet
        if cfg['label_type'] == 'DIR':
            cfg['label_path'] = cfg['train_path']

        train_loader = torch.utils.data.DataLoader(
                                my_dataloader(input_data[0],
                                            cfg['label_type'],
                                            cfg['label_path'],
                                            False,
                                            transforms.Compose([
                                                data_aug_test,
                                                #ImageNetPolicy(),  #autoaug
                                                #Augmentation(fa_resnet50_rimagenet()), #fastaa
                                                transforms.ToTensor(),
                                                my_normalize,
                                                ])),
                                batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

        return train_loader

    elif mode=="val":
        my_dataloader = TensorDatasetTrainClassify
        
        #auto aug

        #from .autoaugment import ImageNetPolicy
        # from libs.FastAutoAugment.data import  Augmentation
        # from libs.FastAutoAugment.archive import fa_resnet50_rimagenet
        if cfg['label_type'] == 'DIR':
            cfg['label_path'] = cfg['val_path']

        data_loader = torch.utils.data.DataLoader(
                                my_dataloader(input_data[0],
                                            cfg['label_type'],
                                            cfg['label_path'],
                                            False,
                                            transforms.Compose([
                                                data_aug_test,
                                                #ImageNetPolicy(),  #autoaug
                                                #Augmentation(fa_resnet50_rimagenet()), #fastaa
                                                transforms.ToTensor(),
                                                my_normalize,
                                                ])),
                                batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

        return data_loader

    elif mode=="eval":
        my_dataloader = TensorDatasetTrainClassify
        
        #auto aug

        #from .autoaugment import ImageNetPolicy
        # from libs.FastAutoAugment.data import  Augmentation
        # from libs.FastAutoAugment.archive import fa_resnet50_rimagenet
        if cfg['label_type'] == 'DIR':
            cfg['label_path'] = cfg['eval_path']

        data_loader = torch.utils.data.DataLoader(
                                my_dataloader(input_data[0],
                                            cfg['label_type'],
                                            cfg['label_path'],
                                            False,
                                            transforms.Compose([
                                                data_aug_test,
                                                #ImageNetPolicy(),  #autoaug
                                                #Augmentation(fa_resnet50_rimagenet()), #fastaa
                                                transforms.ToTensor(),
                                                my_normalize,
                                                ])),
                                batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

        return data_loader
