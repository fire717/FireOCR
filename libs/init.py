import os

from libs._version import __version__
from libs.utils.utils import setRandomSeed, printDash

def initOCR(cfg):

    if cfg["cfg_verbose"]:
        printDash()
        print(cfg)
        printDash()

    print("[INFO] libs verison: "+__version__)


    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['GPU_ID']
    setRandomSeed(cfg['random_seed'])



    if not os.path.exists(cfg['save_dir']):
        os.makedirs(cfg['save_dir'])