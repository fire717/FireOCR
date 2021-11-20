import os
import cv2
import numpy as np
import random





def parseLabel(label_path, img_dir, dict_path):
    
    with open(dict_path,'r', encoding='utf-8') as f:
        lines = f.readlines()
    dict_list = []
    for i,line in enumerate(lines[1:]):
        dict_list.append(line.strip())   
    print(dict_list)


    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    val_ratio = 0.04

    f_train = open('../data/train.txt', 'a', encoding='utf-8')
    f_val = open('../data/val.txt', 'a', encoding='utf-8')

    for line in lines[1:-1]:
        items = line.strip().split(": ")
        img_name = items[0][1:-1]
        label = items[-1][1:-2]
        
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        h,w = img.shape[:2]

        assert h==32

        line = img_name+' '+str(w)+' '+" ".join([str(dict_list.index(x)+1) for x in label])+'\n'

        if random.random()>val_ratio:
            f_train.write(line)
        else:
            f_val.write(line)
    

    f_train.close()
    f_val.close()


def main(img_dir, label_path_list, dict_path):
    
    for label_path in label_path_list:
        parseLabel(label_path, img_dir, dict_path)



if __name__ == '__main__':
    img_dir = "../data/train_resize/"
    label_path_list = ["../data/train/amount/gt.json",
                        "../data/train/date/gt.json"]
    dict_path = "../data/mydict.txt"


    main(img_dir, label_path_list, dict_path)
