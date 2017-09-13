# coding:utf-8

import pandas as pd
import numpy as np
from math import exp
# import matplotlib  
# import matplotlib.pyplot as plt


class LogisticRegression(object):
    '''

    '''

    def __init__(self, data, label):
        self.data = data
        row, column = self.data.shape
        self.row, self.column = row, column
        self.weight = np.zeros([column+1], dtype=np.float64)
        # add new columns
        self.dic = {}
        self.data['b'] = 1.0
        self.dic['b'] = column
        self.data['y'] = label
        self.dic['y'] = column+1
        self.data['f'] = 0.0
        self.dic['f'] = column +2

        for i in xrange(column+1):
            self.data['Li_w%d' % i] = 0.0
        print self.dic

    def train(self):
        
        def calculate_f(row, weight=[]):
            temp_row = row[0:len(weight)]
            temp_weight = np.array(weight)
            # logistic distribute
            return 1.0 /(1.0+ exp(0 - np.sum(temp_row*temp_weight)))
        
        def calculate_Li(row, column):
            #print type(row)
            return (row[self.dic['y']] - row[self.dic['f']]) * row[column]
        def calculate_L(column, rows):
            return np.sum(column) / rows
        j = 1
        index = len(self.weight)
        while( j < 5):
            self.data['f']  = self.data.apply(calculate_f, 1, weight=self.weight)
            for i in xrange(index):
                self.data['Li_w%d' % i] = self.data.apply(calculate_Li, 1, column=i)
            self.weight = self.weight - self.data.ix[:,0-index:].apply(calculate_L, 0, rows=self.row)
            # print "---------- %d-----------\n" % j
            # print self.weight
            j += 1
        
        self.data.info()
        print self.data.head()
    def test(self,data,label):
        def calculate_f(row, weight=[]):
            temp_row = row[0:len(weight)]
            temp_weight = np.array(weight)
            # logistic distribute
            return 1 if 1.0 /(1.0+ exp(0 - np.sum(temp_row*temp_weight))) > 0.5 else -1
        data_temp = data
        data_temp['label'] = label
        data_temp['predict'] = 0
        data_temp['predict'] = data_temp.apply(calculate_f, 1, weight = self.weight)
        print data_temp[ data_temp.label == data_temp.predict ]
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
    lr = LogisticRegression(data=train_data, label=train_label)
    lr.train()
    lr.test(data=test_data, label=test_label)
