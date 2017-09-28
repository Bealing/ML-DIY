import pandas as pd
import numpy as np

from math import exp, log


class StandardScaler(object):
    def __int__(self):
        self.mean = []
        self.var = []
        self.std = []

    def fit(self, data):
        self.data = data
        row, column = self.data.shape
        print row, column
        self.mean = self.data.apply(np.nanmean, 0)
        self.var = self.data.apply(np.nanvar, 0)
        self.std = self.data.apply(np.nanstd, 0)
        print self.mean
        print self.var
        print self.std

        pass


    def transform(self):
        pass

if __name__ == '__main__':
    train_data = '../data/titanic/train.csv'
    test_data = '../data/titanic/test.csv'

    X_train = pd.read_csv(train_data, sep=None)
