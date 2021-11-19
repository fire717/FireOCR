import os,argparse
import random
        
from libs import initOCR, DenseCNN, SwinTransformer, OCRRunner, OCRData

from config import cfg




def main(cfg):


    initOCR(cfg)


    if cfg["model_name"]=='dense':
        model = DenseCNN(cfg["img_size"], cfg["class_number"]+1)
    elif cfg["model_name"]=='swin':
        model = SwinTransformer(img_size=cfg["img_size"], num_classes=cfg["class_number"]+1)
    else:
        raise Exception("Unkown model_name: ", cfg["model_name"])
    
    

    data = OCRData(cfg)
    # data.showTrainData()
    # b
    
    data_loader = data.getTestDataloader()


    runner = OCRRunner(cfg, model)
    runner.modelLoad(cfg['model_path'])

    runner.evaluate(data_loader)



if __name__ == '__main__':
    main(cfg)