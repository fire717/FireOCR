
import os
import random
import numpy as np
from sklearn.model_selection import KFold

import cv2
from torchvision import transforms

from libs.data.datatools import getDataLoader, getFileNames
from libs.data.dataaug_user import TrainDataAug



class OCRData():
    def __init__(self, cfg):
        
        self.cfg = cfg


    def getTrainValDataloader(self):

        # train_img_dir = os.path.join(self.cfg['train_path'], 'img')
        # val_img_dir = os.path.join(self.cfg['val_path'], 'img')
        with open(self.cfg['train_label_path'], 'r') as f:
            train_data = f.readlines()
            train_data.sort(key = lambda x:os.path.basename(x))
            train_data = np.array(train_data)
            random.shuffle(train_data)

        with open(self.cfg['val_label_path'], 'r') as f:
            val_data = f.readlines()

        print("Total train: ",len(train_data)," val: ",len(val_data))


        if self.cfg['try_to_train_items'] > 0:
            train_data = train_data[:self.cfg['try_to_train_items']]
            val_data = val_data[:self.cfg['try_to_train_items']]


        input_data = [train_data, val_data]
        train_loader, val_loader = getDataLoader("trainval", 
                                                input_data,
                                                self.cfg)
        return train_loader, val_loader


    # def getTrainDataloader(self):
    #     data_names = getFileNames(self.cfg['train_path'])
    #     print("[INFO] Total images: ", len(data_names))

    #     input_data = [data_names]
    #     data_loader = getDataLoader("train", 
    #                                     input_data,
    #                                     self.cfg)
    #     return data_loader

    # def getValDataloader(self):
    #     data_names = getFileNames(self.cfg['val_path'])
    #     print("[INFO] Total images: ", len(data_names))

    #     input_data = [data_names]
    #     data_loader = getDataLoader("val", 
    #                                     input_data,
    #                                     self.cfg)
    #     return data_loader

    # def getEvalDataloader(self):
    #     data_names = getFileNames(self.cfg['eval_path'])
    #     print("[INFO] Total images: ", len(data_names))

    #     input_data = [data_names]
    #     data_loader = getDataLoader("eval", 
    #                                     input_data,
    #                                     self.cfg)
    #     return data_loader

    def getTestDataloader(self):
        data_names = getFileNames(self.cfg['test_path'])
        input_data = [data_names]
        data_loader = getDataLoader("test", 
                                    input_data,
                                    self.cfg)
        return data_loader


    def showTrainData(self, show_num = 200):
        #show train data finally to exam

        show_dir = "show_img"
        show_path = os.path.join(self.cfg['save_dir'], show_dir)
        if not os.path.exists(show_path):
            os.makedirs(show_path)


        img_path_list = getFileNames(self.cfg['train_path'])[:show_num]
        transform = transforms.Compose([TrainDataAug(self.cfg['img_size'])])


        for i,img_path in enumerate(img_path_list):
            #print(i)
            img = cv2.imread(img_path)
            img = transform(img)
            img.save(os.path.join(show_path,os.path.basename(img_path)), quality=100)

    