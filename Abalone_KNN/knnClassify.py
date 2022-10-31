#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from math import sqrt
from collections import Counter


# In[ ]:


class knnClassify:
    
    def __init__(self, k):
        self.k = k
        
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return self
    
    def __predict(self, x_test):
        """
        这里计算的是测试集的每个点集到训练集中的每个点集的距离，然后对这些距离进行排序，选取前k个距离最近训练集点集，返回点集索引，根据索引查找         点集的属性，然后对选取的所有点集的属性进行统计投票，其中占比最高的属性就判定为预测点集的属性
        """
        distance = [sqrt(np.sum((data - x_test)**2))for data in self.x_train]
        index = np.argsort(distance)
        top_k = self.y_train[index[:self.k]]
        votes = Counter(top_k).most_common(1)[0][0]
        return votes
    
    def predict(self, x_test):
        y_predict = [self.__predict(x)for x in x_test]
        return y_predict
    
    def score(self, y_true, y_predict):
        score = np.sum(y_true == y_predict)/len(y_true)
        return score

