# coding:utf-8

import pandas as pd
import numpy as np

class LogisticRegression(object):
    '''

    '''

    def __init__(self, data, label):
        self.weight = None
        self.data = data
        self.label = label

    def train(self, data=None, label=None):
        '''
        '''
        def calculate_g(row, weight=[]):
            '''
            temp_row = np.concatenate((row,[1]))
            temp_weight = np.array(weight)
            return np.sum(temp_row*temp_weight)
            '''
            print row

        row, column = data.shape
        self.weight = np.zeros([1,column+1], dtype=np.float64)

        g,f = (0.0,0.0)
        self.data = data.apply(calculate_g,axis=1, weight=self.weight)
        data.info()
        print data.head()

if __name__ == "__main__":
    '''
    '''
    try:
        train_data = pd.read_csv('../data/titanic/titanic_train_data.csv', header=None)
        train_label = pd.read_csv('../data/titanic/titanic_train_label.csv', header=None)
        test_data = pd.read_csv('../data/titanic/titanic_test_data.csv', header=None)
        test_label = pd.read_csv('../data/titanic/titanic_test_label.csv', header=None)
    except IOError, ex:
        print "%s" % ex.message
    ''' 
    train_data.info()
    print train_data.head(10)

    train_label.info()
    print train_label.head(10)
    '''
    lr = LogisticRegression()
    lr.train(data=train_data)

