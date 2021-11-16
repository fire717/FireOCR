import os,argparse
import random
        
from libs import initOCR, DenseCNN, OCRRunner, OCRData

from config import cfg
import pandas as pd



def main(cfg):


    initOCR(cfg)


    model = DenseCNN(cfg)
    
    

    data = OCRData(cfg)
    # data.showTrainData()
    # b
    
    test_loader = data.getTestDataloader()


    runner = OCRRunner(cfg, model)

    #print(model)
    runner.modelLoad(cfg['model_path'])


    res_dict = runner.predict(test_loader)
    print(len(res_dict))
    
    # to csv
    res_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['label'])
    res_df = res_df.reset_index().rename(columns={'index':'image_id'})
    res_df.to_csv(os.path.join(cfg['save_dir'], 'pre.csv'), 
                                index=False,header=True)



if __name__ == '__main__':
    main(cfg)