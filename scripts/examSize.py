import os
import cv2
import numpy as np
#resize h到32后，看看w的分布




def main(read_dir, read_txt):
    with open(read_txt, 'r') as f:
        lines = f.readlines()
    print(read_txt, len(lines))
    for line in lines:
        items = line.strip().split(' ')
        img_name = items[0]
        label_w = int(items[1])
        img = cv2.imread(os.path.join(read_dir, img_name))
        h,w = img.shape[:2]
        if w != label_w:
            print(line, img.shape)

    

if __name__ == '__main__':
    read_dir = "train_all_resize"
    read_txt = "train.txt"
    main(read_dir, read_txt)

    read_txt = "val.txt"
    main(read_dir, read_txt)
