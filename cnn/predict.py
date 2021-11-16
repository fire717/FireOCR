import os,argparse
import random
        
from libs import initOCR, DenseCNN, OCRRunner, OCRData

from config import cfg
import pandas as pd



def main(cfg):


    initOCR(cfg)


    model = DenseCNN(cfg["img_size"], cfg["class_number"]+1)
    
    

    data = OCRData(cfg)
    # data.showTrainData()
    # b
    
    test_loader = data.getTestDataloader()


    runner = OCRRunner(cfg, model)

    #print(model)
    runner.modelLoad(cfg['model_path'])


    res_dict = runner.predict(test_loader)
    print(len(res_dict))
    


if __name__ == '__main__':
    main(cfg)