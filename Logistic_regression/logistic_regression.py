# coding:utf-8

import pandas as pd
import numpy as np
from math import exp, log
# import matplotlib  
# import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression as SKLR
from sklearn.cross_validation import cross_val_score

class LogisticRegression(object):
    '''

    '''

    def __init__(self, data, label):
        self.data = data
        self.threshold = 0.05
        self.step = 10
        row, column = self.data.shape
        self.row, self.column = row, column
        self.weight = np.zeros([column+1], dtype=np.float64)
        # add new columns
        self.dic = {}
        self.data['b'] = 1.0
        self.dic['b'] = column
        self.data['y'] = label
        self.dic['y'] = column+1
        self.data['g'] = 0.0
        self.dic['g'] = column +2

        for i in xrange(column+1):
            self.data['Li_w%d' % i] = 0.0
        print self.dic

    def train(self):
        def calculate_log(t):
            return -log(1 + exp(-t)) if t > 0 else (t - log(1 + exp(t)))
        
        def calculate_f(row, weight=[]):
            temp_row = row[0:len(weight)]
            temp_weight = np.array(weight) 
            # logistic distribute
            return calculate_log(np.sum(temp_row*temp_weight))

        def calculate_Li(row, column):
            #print type(row)
            temp_y = row[self.dic['y']]
            temp_g = row[self.dic['g']]
            return -row[column] / temp_g  if temp_y > 0 else row[column] /(1 - temp_g)
            #return (row[self.dic['y']] - row[self.dic['f']]) * row[column]
        def calculate_L(column, rows):
            return np.sum(column, axis=0) / rows
        
        index = len(self.weight)
        step = np.zeros([index])
        i = 0
        while(True):
            self.data['g']  = self.data.apply(calculate_f, 1, weight=self.weight)
            for i in xrange(index):
                self.data['Li_w%d' % i] = self.data.apply(calculate_Li, 1, column=i)
            temp = self.data.ix[:,0-index:].apply(calculate_L, 0, rows=self.row)
            step = temp - step
            # print step
            if (np.max(step) < self.threshold):
                break
            step = temp
            self.weight = self.weight - step
            i += 1
            print " %d :" % i
            print list(step)
            print list(self.weight)
        
    def test(self,data,label):
        def calculate_f(row, weight=[]):
            temp_row = row[0:len(weight)]
            temp_weight = np.array(weight)
            # logistic distribute
            return 1 if 1.0 /(1.0+ exp(-np.sum(temp_row*temp_weight))) > 0.5 else -1
        data_temp = data
        data_temp['label'] = label
        data_temp['predict'] = 0
        data_temp['predict'] = data_temp.apply(calculate_f, 1, weight = self.weight)
        print data_temp.head()
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
    '''
    sklr = SKLR()
    sklr.fit(train_data, train_label)
    
    res = sklr.predict(test_data)
    test_label.rename(columns={0:'label'}, inplace=True)
    test_label.info()
    print test_label.head()
    test_label['predict'] = 0.0
    test_label['predict'] = res
    print test_label[test_label.label == test_label.predict]
    print sklr.score(test_data,test_label)
    '''
