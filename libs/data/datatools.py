 
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
    def __init__(self, data, img_dir, img_size, load_in_mem= False, transform=None, max_size=45):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
        self.max_size = max_size
        self.load_in_mem = load_in_mem

        if self.load_in_mem:
            self.data_dict = {}
            self.readData()
            print("readData in mem: ", len(self.data_dict))

    def readData(self):
        for i,data in enumerate(self.data):
            print('\r',
                    '{}/{} '.format(
                    i, len(self.data)), 
                    end="",flush=True)

            items = data.strip().split(" ")
            img_name = items[0]
            img = cv2.imread(os.path.join(self.img_dir, img_name))
            self.data_dict[img_name] = img
        print("\n Finish load img to mem.")


    def __getitem__(self, index):
        data = self.data[index]


        items = data.strip().split(" ")
        img_name = items[0]
        label = [int(x) for x in items[1:]]
        if self.load_in_mem:
            img = self.data_dict[img_name].copy()
        else:
            img = cv2.imread(os.path.join(self.img_dir, img_name))
        
        h,w = img.shape[:2]

        if h!=self.img_size[0] or w!=self.img_size[1]:
            imgs = np.ones((self.img_size[0],self.img_size[1],3),dtype=np.uint8)*random.randint(0,255)

            resize_h = self.img_size[0]
            resize_w = int(w*resize_h/h)
            # print(img_name, h,w ,resize_h, resize_w)
            if resize_w>self.img_size[1]:
                print(img_name, h,w ,resize_h, resize_w)
                b
            elif resize_w==self.img_size[1]:
                imgs = cv2.resize(img, (resize_w,resize_h))
            else:
                start_w = random.randint(0,self.img_size[1]-resize_w-1)
                img = cv2.resize(img, (resize_w,resize_h))
                imgs[:, start_w:start_w+resize_w,:] = img
                # cv2.imwrite('t.jpg', imgs)
                # bb
            # imgs = cv2.resize(img, (self.img_size[1],self.img_size[0]))
        else:
            imgs = img

        if self.transform is not None:
            imgs = self.transform(imgs)

        lens = len(label)
        
        if lens > self.max_size:
            print(data, label, len(lines), len(label))
        assert lens <= self.max_size
        # print("lens: ",lens,labels)
        # print(self.max_size,lens,(self.max_size-lens))
        # print([-1]*10)
        # print([-1]*(self.max_size-lens))
        # print(labels+[-1]*(self.max_size-lens))
        #labels = labels+[-1]*(self.max_size-lens)
        # print(imgs.shape, np.array(labels).shape)
        return imgs, np.array(label), lens#, self.data_list[index]
        
    def __len__(self):
        return len(self.data)


class TensorDatasetVal(Dataset):
    _print_times = 0
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform


    def __getitem__(self, index):

        items = self.data[index].strip().split(" ")
        img_name = items[0]
        label = np.array([int(x) for x in items[1:]])

        img = cv2.imread(os.path.join(self.img_dir, img_name))
        # print("000", img.shape)
        img = cv2.resize(img, (350,40))
        # print("111", img.shape)
        if self.transform is not None:
            img = self.transform(img)

        # print("222", img.shape)
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

        img = cv2.resize(img, (350,40))

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
                                            cfg["load_in_mem"],
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
