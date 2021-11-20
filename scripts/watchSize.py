import os
import cv2
import numpy as np
#resize h到32后，看看w的分布


def getAllName(file_dir, tail_list = ['.jpg']): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L

def getW(path):
    img_names = getAllName(path)

    w_list = []
    for img_name in img_names:
        img = cv2.imread(img_name)
        h,w = img.shape[:2]

        new_w = w*32//h
        w_list.append(new_w)
    return w_list

def main(read_path_list):
    total_w_list = []

    for read_path in read_path_list:
        w_list = getW(read_path)
        total_w_list.extend(w_list)

    print("total: ", len(total_w_list))
    print(np.min(total_w_list), np.max(total_w_list), np.mean(np.min(total_w_list),))

if __name__ == '__main__':
    read_path_list = ["../data/train/amount/images",
                    "../data/train/date/images"]


    main(read_path_list)
