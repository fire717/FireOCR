
import os
import torch
import random
import numpy as np


def decodeOne(data):
    #t,c
    res = []
    # print(data)
    for i in range(len(data)):
        idx = np.argmax(data[i])
        # print(idx)
        if len(res)==0:
            res.append(idx)
        else:
            if idx!=res[-1] and idx!=0:
                res.append(idx)
    # print(data,res)
    return np.array(res)

def decodeOutput(output):
    #print(output.shape) #n,t,c
    res = []
    for i in range(len(output)):
        decode_one = decodeOne(output[i])
        res.append(decode_one)
    return res

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
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def printDash(num = 50):
    print(''.join(['-']*num))

