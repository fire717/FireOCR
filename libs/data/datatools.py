 
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



class TensorDatasetTrain(Dataset):
    _print_times = 0
    def __init__(self, data, img_dir, img_size, transform=None, max_size=40):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
        self.max_size = max_size

        self.data_list = []
        self.makeDataGroup()
        # print("train data: ", len(self.data), len(self.data_list))

    def makeDataGroup(self):
        one_batch = []
        now_w = 0
        for i in range(len(self.data)):
            items = self.data[i].strip().split(" ")
            w = int(items[1])
            if now_w==0:
                now_w = w
                one_batch.append(self.data[i])
            elif now_w+w+10>self.img_size[1]:
                #拼接间隔10像素
                self.data_list.append(one_batch)
                one_batch = []
                now_w = 0
            else:
                one_batch.append(self.data[i])
                now_w = now_w+w+10


    def __getitem__(self, index):
        lines = self.data_list[index]
        # print(lines)
        random.shuffle(lines)
        imgs = np.ones((self.img_size[0],self.img_size[1],3),dtype=np.uint8)*128
        labels = []
        start_w = 0
        for line in lines:


            items = line.strip().split(" ")
            img_name = items[0]
            w = int(items[1])
            label = [int(x) for x in items[2:]]

            labels.extend(label)

            img = cv2.imread(os.path.join(self.img_dir, img_name))
            #print(w,img.shape,start_w)
            imgs[:, start_w:start_w+w,:] = img
            start_w = start_w+w+10
            # cv2.imwrite('t1.jpg', img)

        # cv2.imwrite('t.jpg', imgs)
        # print(labels)
        # b
        if self.transform is not None:
            imgs = self.transform(imgs)

        lens = len(labels)
        assert lens <= self.max_size
        # print("lens: ",lens,labels)
        # print(self.max_size,lens,(self.max_size-lens))
        # print([-1]*10)
        # print([-1]*(self.max_size-lens))
        # print(labels+[-1]*(self.max_size-lens))
        #labels = labels+[-1]*(self.max_size-lens)
        # print(imgs.shape, np.array(labels).shape)
        return imgs, np.array(labels), lens#, self.data_list[index]
        
    def __len__(self):
        return len(self.data_list)


class TensorDatasetVal(Dataset):
    _print_times = 0
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform


    def __getitem__(self, index):

        items = self.data[index].strip().split(" ")
        img_name = items[0]
        label = np.array([int(x) for x in items[2:]])

        img = cv2.imread(os.path.join(self.img_dir, img_name))

        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.data[index]
        
    def __len__(self):
        return len(self.data)

class TensorDatasetTest(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):

        img = cv2.imread(self.data[index])

        if self.transform is not None:
            img = self.transform(img)


        return img, self.data[index]

    def __len__(self):
        return len(self.data)


###### 3. get data loader 



def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    # print(batch)
    # print(batch[0])
    imgs = torch.stack(batch[0])
    # print(imgs)
    # b
    labels = torch.from_numpy(np.concatenate(batch[1], 0))
    lens = torch.from_numpy(np.array(batch[2]))
    # line = batch[3]
    # print(labels, lens)
    # print(imgs.shape, labels.shape, lens.shape)
    del batch
    return imgs, labels, lens


def getDataLoader(mode, input_data, cfg):




    data_aug_train = TrainDataAug(cfg['img_size'])
    data_aug_test = TestDataAug(cfg['img_size'])




    if mode=="pre":
        my_dataloader = TensorDatasetTest

        test_loader = torch.utils.data.DataLoader(
                my_dataloader(input_data[0],
                                transforms.Compose([
                                    data_aug_test,
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5,1)
                                ])
                ), batch_size=1, shuffle=False, 
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
            )

        return test_loader


    elif mode=="trainval":

        img_dir = cfg['img_dir']

        train_loader = torch.utils.data.DataLoader(
                                        TensorDatasetTrain(input_data[0],
                                            img_dir,
                                            cfg['img_size'], 
                                            transforms.Compose([
                                                data_aug_train,
                                                transforms.ToTensor(),
                                                transforms.Normalize(0.5,1)
                                        ])),
                                batch_size=cfg['batch_size'], 
                                shuffle=True, 
                                num_workers=cfg['num_workers'], 
                                pin_memory=cfg['pin_memory'],
                                collate_fn=collate_fn,
                                )#prefetch_factor=4

        val_loader = torch.utils.data.DataLoader(
                                        TensorDatasetVal(input_data[1],
                                            img_dir,
                                            transforms.Compose([
                                                data_aug_test,
                                                transforms.ToTensor(),
                                                transforms.Normalize(0.5,1)
                                        ])),
                                batch_size=1, 
                                shuffle=False, 
                                num_workers=cfg['num_workers'], 
                                pin_memory=cfg['pin_memory'],
                                )#prefetch_factor=4
        return train_loader, val_loader



    elif mode=="test":
        my_dataloader = TensorDatasetTrainVal

        img_dir = cfg['test_img_dir']


        data_loader = torch.utils.data.DataLoader(
                                        my_dataloader(input_data[0],
                                            img_dir,
                                            transforms.Compose([
                                                data_aug_test,
                                                transforms.ToTensor(),
                                                transforms.Normalize(0.5,1)
                                        ])),
                                batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

        return data_loader
