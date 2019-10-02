#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:06:45 2019

@author: root
"""

import numpy as np


def split_data(data, proportion):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array, to be split along the first axis
        - proportion: a float less than 1
    """
    size = data.shape[0]
    split_idx = int(proportion * size)
    np.random.shuffle(data)
    return data[:split_idx], data[split_idx:]

def one_hot_encode(data,num_classes):
    data = np.array(data,dtype=np.int8).reshape(-1)
    print(data.shape)
    
    return np.eye(num_classes)[data]


def train_test_split(x,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    return X_train, X_test, y_train, y_test

