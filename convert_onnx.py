import os,argparse
import random
import torch        
from libs import *

from config import cfg
import pandas as pd



def main(cfg):


    initOCR(cfg)


    if cfg["model_name"]=='dense':
        model = DenseCNN(cfg["img_size"], cfg["class_number"]+1)
    elif cfg["model_name"]=='mobilenetv2':
        model = MobileNetV2(cfg["class_number"]+1)
    elif cfg["model_name"]=='swin':
        model = SwinTransformer(img_size=cfg["img_size"], num_classes=cfg["class_number"]+1)
    else:
        raise Exception("Unkown model_name: ", cfg["model_name"])
    
    

    # data = OCRData(cfg)
    # # data.showTrainData()
    # # b
    
    # test_loader = data.getTestDataloader()


    runner = OCRRunner(cfg, model)

    #print(model)
    runner.modelLoad("output/mobilenetv2_e10_0.92221.pth")


    runner.model.eval()
    runner.model.to("cuda")

    #data type nchw
    dummy_input1 = torch.randn(1, 1, 40, 350).to("cuda")
    input_name = "input1" #自己命名
    output_name = "output1"
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(model, dummy_input1, "output/model.onnx", 
        verbose=True, input_names=[input_name], output_names=[output_name],
        dynamic_axes= {
                        input_name: {0: 'batch_size', 3 : 'in_width'}}
                  ,
        do_constant_folding=True)




if __name__ == '__main__':
    main(cfg)