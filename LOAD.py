#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 04:45:16 2018

@author: polo
"""
#_______________________________________________________________
import numpy as np
import pandas as pd

Train_labels = []
Train_sequences = []
    
Test_labels = []
Test_sequences = []

def LOAD_DATASET():
    
    df = pd.read_csv('/home/polo/.config/spyder-py3/TweetAgeEstimation/POS_CODED.tsv',delimiter='\t',encoding='utf-8')
    
    i = 0
    for Tags in df.Sequence.str.split():
        Tag_C = np.zeros((80,20))
        for j in range(len(Tags)):
            Tag_C[j] = np.asarray(np.asarray([int(i) for i in list(Tags[j])]))
        if i<8000:
           Train_sequences.append(Tag_C)
        else:
            Test_sequences.append(Tag_C)   
        i+=1
    i = 0
    for Cls in df.Class.str.replace(' ',''):
        label = [int(i) for i in list(Cls)]
        label = np.asarray(label)
        label = label[::-1]

        if i<8000: 
           Train_labels.append(label)
        else:
            Test_labels.append(label)
        i+=1
LOAD_DATASET()

#Tag_C = Tag_C.reshape(80*20)
