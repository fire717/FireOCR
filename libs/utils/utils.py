
import os
import torch
import random
import numpy as np


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
        softmax_v = softmax(data[i])
        # print(softmax_v)
        score = np.max(softmax_v)
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
    return np.array(res), score
    
def decodeOutput(output):
    #print(output.shape) #n,t,c
    res = []
    confs = []
    for i in range(len(output)):
        idx, scores = decodeOne(output[i])
        res.append(idx)
        confs.append(scores)
    return res, confs

def setRandomSeed(seed=42):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    setRandomSeed(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def printDash(num = 50):
    print(''.join(['-']*num))

