#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: Panos,Yazid
"""
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn

from matplotlib import pyplot as plt

##_______________________Parameters_______________________

#Number of Iterations
nb_epochs = 100
#size of batch
batch_size = 8000
#number of Classes
n_classes= 8
#unrolled through 80 time steps
time_steps = 80
#rows of 20 POS
n_input = 20
#hidden LSTM units
num_units = 8
#learning rate for adam
learning_rate = 0.02
dropout_keep_prob = 0.1

#_______________________________________________________________

Train_labels = []
Train_sequences = []
    
Test_labels = []
Test_sequences = []

#_______________________________________________________________

def LOAD_DATASET():
    
    df = pd.read_csv('/home/polo/.config/spyder-py3/TweetAgeEstimation/POS_CODED.tsv',delimiter='\t',encoding='utf-8')
    
    i = 0
    for Tags in df.Sequence.str.split():
        Tag_C = np.zeros((80,20))
        for j in range(len(Tags)):
            Tag_C[j] = np.asarray(np.asarray([int(i) for i in list(Tags[j])]))[::-1]
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
    
    #return Train_sequences,Train_labels,Test_sequences,Test_labels
    return np.array(Train_sequences),np.array(Train_labels),np.array(Test_sequences),np.array(Test_labels)


#_______________________________________________________________

Train_sequences,Train_labels,Test_sequences,Test_labels = LOAD_DATASET()

print('_______________________________________________________________')

dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

#weights and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random_normal([num_units,n_classes]),name='out_weights')
out_bias = tf.Variable(tf.random_normal([n_classes]),name='out_bias')

x = tf.placeholder('float', [None, time_steps,n_input],name='X')
y = tf.placeholder("string", [None, n_classes], name='Y')

#_______________________________________________________________

input_X = tf.unstack(x ,time_steps,1)

#_______________________________________________________________

def LSTM_NN():
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
         lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1.0)
         outputs,_ = rnn.static_rnn(lstm_layer, input_X, dtype="float32")
    outputs = tf.matmul(outputs[-1], out_weights) + out_bias
    return outputs

#_______________________________________________________________

#lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1.0)
#outputs,_ = rnn.static_rnn(lstm_layer, input_X, dtype="float32")
#outputs = tf.matmul(outputs[-1], out_weights) + out_bias

#logits = outputs
logits = LSTM_NN()

prediction = logits

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = y))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))

correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')
         
optimization = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#_______________________________________________________________

train_loss_vec = []
train_acc_vec = []

test_acc_vec = []

#_______________________________________________________________

saver = tf.train.Saver()

#_______________________________________________________________

init=tf.global_variables_initializer()
with tf.Session() as sess:
     sess.run(init)
     
     #batch_x = np.asarray(Train_sequences).reshape((-1, time_steps, n_input))
     #batch_y =  np.asarray(Train_labels)#.resh
     
     #batch_x = Train_sequences
     batch_x = np.asarray(Train_sequences).reshape((-1, time_steps, n_input))

     batch_y = np.asarray(Train_labels)#.resh
     
     Tbatch_x = Test_sequences
     Tbatch_y = np.asarray(Test_labels)#.resh
     
     for iter in range(1,nb_epochs+1):
         sess.run(optimization, feed_dict={x: batch_x, y: batch_y})
                
         acc = sess.run(accuracy,feed_dict={x: batch_x, y: batch_y})             
         los = sess.run(loss,feed_dict={x: batch_x, y: batch_y})
         if iter%5 == 0:
            print('_________',iter,'_________')

            print("Train Accuracy ",acc)
            print("Train Loss ",los)

            train_loss_vec.append(los)
            train_acc_vec.append(acc)
       
            print('__________________TEST____________________')
            acc = sess.run(accuracy, feed_dict={x: Tbatch_x, y: Tbatch_y})
            print("Testing Accuracy:", acc)
            print("Testing Loss:", sess.run(loss,feed_dict={x: Tbatch_x, y: Tbatch_y}))
            
            test_acc_vec.append(acc)

#_______________________________________________________________
     
def Plot_Loss_accuracy():
    plt.plot(train_loss_vec, 'k-', lw=2, label='Train Loss')
    plt.plot(train_acc_vec, 'r:', label='Train Accuracy')

    plt.plot(test_acc_vec, 'b:', label='Test Accuracy')

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy and Loss')
    plt.title('Accuracy and Loss LSTM NN')
    plt.grid()
    plt.legend(loc='upper left')#plt.legend(loc='lower right')
    plt.show()

Plot_Loss_accuracy()