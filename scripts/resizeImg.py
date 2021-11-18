import os
import cv2
import numpy as np
#resize h到32后，看看w的分布




def getW(path):
    img_names = getAllName(path)

    w_list = []
    for img_name in img_names:
        img = cv2.imread(img_name)
        h,w = img.shape[:2]

        new_w = w*32//h
        w_list.append(new_w)
    return w_list

def main(read_dir, save_dir, new_h = 32):
    img_names = os.listdir(read_dir)

    for img_name in img_names:
        read_path = os.path.join(read_dir, img_name)
        img = cv2.imread(read_path)
        h,w = img.shape[:2]

        new_w = w*new_h//h+1

        new_img = cv2.resize(img, (new_w, new_h))
        save_path = os.path.join(save_dir, img_name)

        cv2.imwrite(save_path, new_img)


if __name__ == '__main__':
    read_dir = 'train_all'
    save_dir = 'train_all_resize'


    main(read_dir, save_dir)
