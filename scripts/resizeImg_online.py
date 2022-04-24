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



def main(read_dir, save_dir, new_h):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_names = getAllName(read_dir)

    for read_path in img_names:
        img = cv2.imread(read_path)
        h,w = img.shape[:2]

        new_w = w*new_h//h+1

        new_img = cv2.resize(img, (new_w, new_h))
        save_path = os.path.join(save_dir, os.path.basename(read_path))

        cv2.imwrite(save_path, new_img)


if __name__ == '__main__':
    read_dir = '/aiwin/data/imgs'
    save_dir = '/aiwin/data/imgs_resize'
    new_h = 32


    main(read_dir, save_dir, new_h)
