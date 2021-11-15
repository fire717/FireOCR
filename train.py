import os,argparse
import random
        
from libs import initOCR, DenseCNN, OCRRunner, OCRData

from config import cfg




def main(cfg):


    initOCR(cfg)


    model = DenseCNN(cfg["img_size"], cfg["class_number"]+1)
    #print(model)

    data = OCRData(cfg)
    # data.showTrainData()
    # b
    
    train_loader, val_loader = data.getTrainValDataloader()


    runner = OCRRunner(cfg, model)
    runner.train(train_loader, val_loader)


    ## test
    #test_loader = data.getTestDataloader()
    #runner.test(test_loader)


if __name__ == '__main__':
    main(cfg)