#coding:utf-8
# from __future__ import print_function
import numpy as np
import cv2
# from cv2 import dnn
# import sys
 
# import tensorflow as tf
# from tensorflow.python.framework import graph_util
# import os

# import time

import os
import time

import onnxruntime as rt
# import keys

# def decode(pred):
#     char_list = []
#     pred_text = pred.argmax(axis=2)[0]
#     for i in range(len(pred_text)):
#         if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
#             char_list.append(characters[pred_text[i]])
#     return u''.join(char_list)
alphabet = {}
with open("char_QRCode.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i,line in enumerate(lines):
    alphabet[i] = line.strip()
print("load dict: ", len(alphabet))


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def decodeOne(data):

    #t,c
    res = []
    scores = []
    # print(data)
    #print("--------------")
    last_idx = -1

    for i in range(len(data)):
        idx = np.argmax(data[i])
        # print(data[i])
        # softmax_v = softmax(data[i])
        # print(softmax_v)
        score = np.max(data[i])
        #print(idx)
        if len(res)==0:
            if idx!=0:
                res.append(idx)
                scores.append(score)
        else:
            if idx!=last_idx and idx!=0:
                res.append(idx)
                scores.append(score)
        last_idx = idx
    #print(res)
    score = 0
    if len(scores)>0:
        score = np.mean(scores)

    chars = ''.join([alphabet[x] for x in res])
    return chars,np.array(res), score


# characters = keys.alphabet[:]
# characters = characters[1:] + u'卍'
# nclass = len(characters)



model_path = 'model.onnx'
sess=rt.InferenceSession(model_path)#model_path就是模型的地址
input_name=sess.get_inputs()[0].name

names = os.listdir("imgs")
print("total: ", len(names))

for name in names:
    img = cv2.imread(os.path.join("imgs", name))
    # print("img shape: ", img.shape)
    # img = cv2.resize(img, ( 350, 40))
    resize_h = 40
    h,w = img.shape[:2]
    resize_w = int(w*resize_h/h)#+1
    img = cv2.resize(img, (resize_w,resize_h))
    #img = img[:, :, [ 2, 1, 0]] # BGR2RGB
    # print("img shape: ", img.shape)

    if len(img.shape)==3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # data = img.reshape( 1, img.shape[ 0], img.shape[ 1], 1) #keras
    data = img.reshape( 1, 1, img.shape[ 0], img.shape[ 1])  #pth

    # print(data[0][0][:,0])
    # data = np.array([[img.shape[ 0], img.shape[ 1]]])
    #print(data.shape)
    #data = np.transpose(data,(0,3,1,2))
    # print(data.shape)
    #data = data/255.0 - 0.5 #keras
    data = data/255.0 - 0.5
    #print(data.shape)
    # print(data[0][0][:,0])
    data = data.astype(np.float32)
    # print(data.shape)
    # print(data)
    # b
    for _ in range(10):
        t = time.time()
        res=sess.run(None,{input_name:data})[0]
        # print(res)
        # print(np.sum(res))
        # b
        print(time.time() - t)

    # print("res: ", res[0][:20])
    # print("res: ", np.array(res).shape)


    out = decodeOne(res[0])
    # print(out)
    
    save_txt = os.path.join("imgs", name[:-3]+"txt")
    with open(save_txt, 'w', encoding='utf-8') as f:
        f.write(out[0])
