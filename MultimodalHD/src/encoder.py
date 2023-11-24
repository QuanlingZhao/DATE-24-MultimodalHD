import torch
import numpy as np
import os
import math
import random
import copy
from scipy.sparse import random
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt




class Encoder():
    def __init__(self,configs,min,max,range):
        self.configs = configs

        self.mode = configs['config']['dataset']
        self.quantization_num = int(configs['config']['quantization_num'])
        self.D = int(configs['config']['D'])
        self.P = float(configs['config']['P'])
        self.min = min
        self.max = max
        self.range = range
        self.init_hvs()


    def init_hvs(self):
        # level hvs
        num_flip = int(self.D * self.P)
        self.level_hvs = [np.random.randint(2, size=self.D)]
        for i in range(self.quantization_num-1):
            new = copy.deepcopy(self.level_hvs[-1])
            idx = np.random.choice(self.D,num_flip,replace=False)
            new[idx] = 1-new[idx]
            self.level_hvs.append(new)
        self.level_hvs = np.stack(self.level_hvs)

        #id hvs
        self.id_hvs = []
        if self.mode == "HAR":
            for i in range(9):
                self.id_hvs.append(np.random.randint(2, size=self.D))
        if self.mode == "MHEALTH":
            for i in range(21):
                self.id_hvs.append(np.random.randint(2, size=self.D))
        if self.mode == "OPP":
            for i in range(39):
                self.id_hvs.append(np.random.randint(2, size=self.D))
        self.id_hvs = np.stack(self.id_hvs)


    def quantize(self, one_sample):
        T,M = one_sample.shape
        quantization = self.level_hvs[((((one_sample - self.min) / self.range) * self.quantization_num) - 1).astype('i')]
        return quantization

    def bind(self,a,b):
        return np.logical_xor(a,b).astype('i')

    def permute(self,a):
        for i in range(len(a)):
            a[i] = np.roll(a[i],i,axis=1)
        return a

    def sequential_bind(self,a):
        return np.sum(a,axis=0) % 2

    def bipolarize(self,a):
        a[a==0] = -1
        return a



    def encode_one_sample(self,one_sample):
        T = len(one_sample)
        out = self.quantize(one_sample)
        out = self.bind(out,np.repeat(np.expand_dims(self.id_hvs,0),T,0))
        out = self.permute(out)
        out = self.sequential_bind(out)
        out = self.bipolarize(out).astype(np.int16)
        return out
    





