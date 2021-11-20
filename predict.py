import os,argparse
import random
        
from libs import initOCR, DenseCNN, OCRRunner, OCRData

from config import cfg
import pandas as pd
import json


def main(cfg):


    initOCR(cfg)


    if cfg["model_name"]=='dense':
        model = DenseCNN(cfg["img_size"], cfg["class_number"]+1)
    # elif cfg["model_name"]=='swin':
    #     model = SwinTransformer(img_size=cfg["img_size"], num_classes=cfg["class_number"]+1)
    else:
        raise Exception("Unkown model_name: ", cfg["model_name"])
    
    

    data = OCRData(cfg)
    # data.showTrainData()
    # b
    
    test_loader = data.getPreDataloader()


    runner = OCRRunner(cfg, model)

    #print(model)
    runner.modelLoad(cfg['model_path'], False)


    res_dict = runner.predict(test_loader)
    print(len(res_dict))
    
    with open(r"answer.json",'w') as f:  
        json.dump(res_dict, f, indent=2, ensure_ascii=False)     
    print('done')

if __name__ == '__main__':
    main(cfg)