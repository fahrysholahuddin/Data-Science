import numpy as np
import pandas as pd

class class_missingValue:
    X = []
    y = []
    a = []
    b = []
    name = ''

    def __init__(self, data, pick=None):
        self.X = data.X
        self.y = data.y
        self.a = np.isnan(data.X)
        self.name = pick
        self.pick()
        #self.b = np.isnan(data.y)

    def pick(self):
        if self.name == "mean":
            self.mean()
        elif self.name == "median":
            self.median()

    def mean(self):
        print('Data Missing Value Diatasi Dengan Mean')
        print("data kosong di\n",self.a)
        self.X[self.a] = np.nanmean(self.X)
        print("data tersebut diisi dengan nilai : ",self.X[self.a])
         

    def median(self):
        print('Data Missing Value Diatasi Dengan Median')
        print("data kosong di\n",self.a)
        self.X[self.a] = np.nanmedian(self.X)
        print("data tersebut diisi dengan nilai : ",self.X[self.a])