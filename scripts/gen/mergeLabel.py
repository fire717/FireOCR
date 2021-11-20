import os
import cv2
import numpy as np
import random



lines = []
with open('../../../data/challange/train_balanced.txt', 'r') as f:
    lines.extend(f.readlines())
with open('dataset/gen_train.txt', 'r') as f:
    lines.extend(f.readlines())

random.shuffle(lines)
with open('../../../data/challange/train_bal_add_gen.txt', 'w') as f:
    for line in lines:
        f.write(line)



lines = []
with open('../../../data/challange/val_balanced.txt', 'r') as f:
    lines.extend(f.readlines())
with open('dataset/gen_val.txt', 'r') as f:
    lines.extend(f.readlines())

random.shuffle(lines)
with open('../../../data/challange/val_bal_add_gen.txt', 'w') as f:
    for line in lines:
        f.write(line)