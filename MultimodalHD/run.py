import configparser
import sys
from src.server import *
import torch
import numpy as np
import random






if __name__ == '__main__':
    configfile = sys.argv[1]
    parser = configparser.ConfigParser()
    print("================Configs================")
    parser.read(configfile)
    for sect in parser.sections():
       print('Section:', sect)
       for k,v in parser.items(sect):
          print(' {} = {}'.format(k,v))
    configs = parser
    print("=======================================")
    assert configs['config']['dataset'] in ['HAR','MHEALTH','OPP']

    seed = int(configs['config']['seed'])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    FLServer = Server(configs)

    FLServer.start()

