#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:39:48 2019

@author: root
"""


import util
import model
from sklearn.metrics import accuracy_score
import math
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os
import sys
import seaborn as sn

tf.reset_default_graph()

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)
   
flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/suresh', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '/home/suresh/fmnist', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('max_epoch_num', 300, '')
FLAGS = flags.FLAGS

# load data
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

y_train= util.one_hot_encode(y_train,10)
y_test = util.one_hot_encode(y_test,10)

#The following line can be run if we want to get a separate validation set
#X_train, X_validation, y_train, y_validation = util.train_test_split(x_train, y_train)

def train(X_train, y_train):   
    train_num_examples = X_train.shape[0]
    
    ce_train = []
    y_train_predicted = []
    for i in range(math.ceil(train_num_examples / batch_size)):
        batch_xs = X_train[i*batch_size:(i+1)*batch_size, :]
        batch_ys = y_train[i*batch_size:(i+1)*batch_size, :]       
        _, ce, y_predicted = session.run([train_op, cross_entropy, output], {x: batch_xs, y: batch_ys}) 
        
        y_train_predicted.append(y_predicted)
        ce_train.append(ce)
       
        
    avg_ce_train = np.mean(np.concatenate(ce_train).ravel())
    train_accuracy = accuracy_score(np.argmax(y_train, axis=1), np.argmax(np.vstack(y_train_predicted), axis=1))
   
    print('TRAIN CROSS ENTROPY: ' + str(avg_ce_train))
    print("TRAIN ACCURACY:",train_accuracy)
   
    return avg_ce_train, train_accuracy


def validate(X_validation, y_validation):
    
    validation_num_examples = X_validation.shape[0]
    ce_validation = []
    y_validation_predicted = []
    for i in range(math.ceil(validation_num_examples / batch_size )):
        batch_xs = X_validation[i*batch_size:(i+1)*batch_size, :]
        batch_ys = y_validation[i*batch_size:(i+1)*batch_size, :]
        ce, y_predicted = session.run([cross_entropy, output], {x: batch_xs, y: batch_ys})
        ce_validation.append(ce)
        y_validation_predicted.append(y_predicted)

    avg_ce_validation = np.mean(np.concatenate(ce_validation).ravel())
    validation_accuracy = accuracy_score(np.argmax(y_validation, axis=1), np.argmax(np.vstack(y_validation_predicted), axis=1))
   
    print('VALIDATION CROSS ENTROPY: ' + str(avg_ce_validation))
    print('VALIDATION ACCURACY:', validation_accuracy)
    print('\n')
    
    return avg_ce_validation, validation_accuracy


def test(X_test, y_test):   
    test_num_examples = X_test.shape[0]
  
    ce_test = []
    y_test_predicted = []
    for i in range(math.ceil(test_num_examples / batch_size)):
        batch_xs = X_test[i*batch_size:(i+1)*batch_size, :]
        batch_ys = y_test[i*batch_size:(i+1)*batch_size, :]
        
        ce, y_predicted = session.run([cross_entropy, output], {x: batch_xs, y: batch_ys})
        ce_test.append(ce)
        y_test_predicted.append(y_predicted)
        
    avg_ce_test = np.mean(np.concatenate(ce_test).ravel())
    test_accuracy =  accuracy_score(np.argmax(y_test, axis=1), np.argmax(np.vstack(y_test_predicted), axis=1))
        
    print('TEST CROSS ENTROPY: ' + str(avg_ce_test))
    print('TEST ACCURACY:', test_accuracy)
    
    cm = tf.confusion_matrix(labels=np.argmax(y_test,axis=1), predictions=np.argmax(np.vstack(y_test_predicted),axis=1), num_classes=10)
    print("Confusion matrix is: \n ")
    conf_mat = session.run(cm)
    return avg_ce_test, test_accuracy, conf_mat

  

# run training
batch_size = FLAGS.batch_size
mean_validation_loss = []

results = []
    
tf.reset_default_graph()

if len(sys.argv) == 1:
    x,y,output = model.create_architecture_1()

if len(sys.argv) > 1:
    if sys.argv[1] == 'architecture_1':
        x,y,output = model.create_architecture_1()
        
    elif sys.argv[1] == 'architecture_1_with_regularization':
        x,y,output = model.create_architecture_1_with_regularization()
        
    elif sys.argv[1] == 'architecture_2':
        x,y,output = model.create_architecture_2()
        
    elif sys.argv[1] == 'architecture_2_with_regularization':
        x,y,output = model.create_architecture_2_with_regularization()
        
        
    else:
        
        print("Argument error")
        exit()



cross_entropy  = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output, name='cross_entropy')

regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# this is the weight of the regularization part of the final loss
REG_COEFF = 0.1
# this value is what we'll pass to `minimize`
if regularization_losses:
    total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)
    print("regularization used")
else:
    total_loss = cross_entropy
    
confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)

# set up training and saving functionality
global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
saver = tf.train.Saver()

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)

ce_validation_per_epoch = []
ce_train_per_epoch = []

train_history = []
validation_history = []

best_cost = 1000
no_improvement = 0

with tf.Session() as session:
    session.run(tf.global_variables_initializer()) 
    for epoch in range(FLAGS.max_epoch_num):
        print('Epoch: ' + str(epoch) )
        
        train_loss, train_accuracy = train(x_train, y_train)
        ce_train_per_epoch.append(train_loss)
        train_history.append(train_accuracy)
        
        validation_loss, validation_accuracy = validate(x_test, y_test)
        ce_validation_per_epoch.append(validation_loss)
        validation_history.append(validation_accuracy)
             
        if validation_loss <= best_cost:
            best_cost = validation_loss
            saved_path = saver.save(session, os.path.join(FLAGS.save_dir, "model_1"), global_step=global_step_tensor)
            no_improvement = 0
        else:
            no_improvement = no_improvement + 1
            
        if no_improvement > 30:
            print("Exiting due to overfitting......")
            break           
        
    test_loss, test_accuracy, cm = test(x_test, y_test)
    normalized_cm = np.round(cm/y_test.shape[0]*100,0)
    
#The following code creates normalized confusion matrix using seaborn package.  

    fig = plt.figure()
    fig.tight_layout()
    x_axis_labels = ['Tshirt','Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt','Sneaker', 'Bag', 'Ankle boot']
    y_axis_labels = ['Tshirt','Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt','Sneaker', 'Bag', 'Ankle boot']
    ax = sn.heatmap(normalized_cm, ax= plt.axes(), annot=True,xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    ax.set_ylim(ax.get_ylim()[0]+0.5,ax.get_ylim()[1]-0.5,)
    
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.gcf().subplots_adjust(left=0.20)
    plt.title('Confusion matrix')
    figure = ax.get_figure()
    figure.savefig('cm.png', dpi=300) 

  
# Generate Loss Plot
plt.clf()     
plt.figure(figsize=(10,6))
plt.plot(ce_train_per_epoch, label = 'train loss')
plt.plot(ce_validation_per_epoch, label = 'test loss')
plt.legend(loc = 'upper left')
plt.title('Train and Test Loss')
plt.grid()
plt.savefig('loss.png', dpi=300)
plt.show()

#Generate Accuracy PLot    
plt.figure(figsize=(10,6))
plt.plot(train_history, label = 'train accuracy')
plt.plot(validation_history, label = 'test accuracy')
plt.legend(loc = 'upper left')
plt.title('Train and Test Accuracy')
plt.grid()
plt.savefig('accuracy.png', dpi=300)
plt.show()


# Calculate Confidence Interval
error = 1 - test_accuracy
conf_interval_upper = error + 1.96*math.sqrt((error*(1-error))/y_test.shape[0])
conf_interval_lower = error - 1.96*math.sqrt((error*(1-error))/y_test.shape[0])

print('upper_bound' + str(conf_interval_upper))
print('lower_bound' + str(conf_interval_lower))
