import os,argparse
import random
import torch        
from libs import initOCR, DenseCNN, OCRRunner, OCRData

from config import cfg
import pandas as pd



def main(cfg):


    initOCR(cfg)


    model = DenseCNN(cfg["img_size"], cfg["class_number"]+1)
    



    runner = OCRRunner(cfg, model)

    #print(model)
    # runner.modelLoad(cfg['model_path'])


    runner.model.eval()
    runner.model.to("cuda")


    #data type nchw
    dummy_input1 = torch.randn(1, 1, 32, 280).to("cuda")
    input_names = [ "input1"] #自己命名
    output_names = [ "output1"]
    
    torch.onnx.export(model, dummy_input1, "output/model.onnx", 
        verbose=True, input_names=input_names, output_names=output_names,
        do_constant_folding=True)




if __name__ == '__main__':
    main(cfg)