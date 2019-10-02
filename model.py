#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:04:49 2019

@author: root
"""
import tensorflow as tf

# specify the network


def create_architecture_1():
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    x = x/255.0
    
    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(x, 100, activation=tf.nn.leaky_relu,
                                 name='hidden_layer_1')
        
        
        hidden_2 = tf.layers.dense(hidden_1,100,activation=tf.nn.leaky_relu,
                                  name='hidden_layer_2')
        
        output = tf.layers.dense(hidden_2, 10, name='output_layer')
        
    tf.identity(output, name='output')

    y = tf.placeholder(tf.float32, [None, 10], name='label')
    
    return x,y,output

def create_architecture_1_with_regularization():
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    x = x/255.0
    
    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(x, 300, activation=tf.nn.leaky_relu,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                 name='hidden_layer_1')
        
        dropout_1 = tf.layers.dropout(hidden_1,rate=0.5,name='dropout_1')
        
        hidden_2 = tf.layers.dense(dropout_1,200,activation=tf.nn.leaky_relu,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                 name='hidden_layer_2')
        
        
        output = tf.layers.dense(hidden_2, 10, name='output_layer')
        
    tf.identity(output, name='output')

    y = tf.placeholder(tf.float32, [None, 10], name='label')
    
    return x,y,output

def create_architecture_2():
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 784], name='inpput_placeholder')
    x = x/255.0
    
    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(x, 200, activation=tf.nn.leaky_relu,
                                 name='hidden_layer_1')
      
        hidden_2 = tf.layers.dense(hidden_1,200,activation=tf.nn.leaky_relu,
                                  name='hidden_layer_2')
        
        hidden_3 = tf.layers.dense(hidden_2, 200, activation=tf.nn.leaky_relu, 
                                  name='hidden_layer_3')
        
        hidden_4 = tf.layers.dense(hidden_3, 200, activation=tf.nn.leaky_relu,                                 
                                  name='hidden_layer_4')            
        
        hidden_5 = tf.layers.dense(hidden_4, 200, activation=tf.nn.leaky_relu, 
                                  name='hidden_layer_5')
        
        hidden_6 = tf.layers.dense(hidden_5, 200, activation=tf.nn.leaky_relu, 
                                  name='hidden_layer_6')
        
        hidden_7 = tf.layers.dense(hidden_6, 200, activation=tf.nn.leaky_relu, 
                                  name='hidden_layer_7')
        
        
        hidden_8 = tf.layers.dense(hidden_7, 200, activation=tf.nn.leaky_relu, 
                                  name='hidden_layer_8')
        
        output = tf.layers.dense(hidden_8, 10, name='output_layer')
        
    tf.identity(output, name='output')
        
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    
    return x,y,output


def create_architecture_2_with_regularization():
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    x = x/255.0
    
    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(x, 100, activation=tf.nn.leaky_relu,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                 name='hidden_layer_1')
        
        dropout_1 = tf.layers.dropout(hidden_1,rate=0.5,name='dropout_1')
        
        hidden_2 = tf.layers.dense(dropout_1,100,activation=tf.nn.leaky_relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), 
                                  name='hidden_layer_2')
        
        dropout_2 = tf.layers.dropout(hidden_2,rate=0.5, name='dropout_2')
        
        hidden_3 = tf.layers.dense(dropout_2, 100, activation=tf.nn.leaky_relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name='hidden_layer_3')
        
        dropout_3 = tf.layers.dropout(hidden_3,rate=0.5,name='dropout_3')
        
        hidden_4 = tf.layers.dense(dropout_3, 100, activation=tf.nn.leaky_relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name='hidden_layer_4')
        
        dropout_4 = tf.layers.dropout(hidden_4,rate=0.5,name='dropout_4')
        
        hidden_5 = tf.layers.dense(dropout_4, 100, activation=tf.nn.leaky_relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name='hidden_layer_5')
        
        dropout_5 = tf.layers.dropout(hidden_5,rate=0.5,name='dropout_5')
        
        hidden_6 = tf.layers.dense(dropout_5, 100, activation=tf.nn.leaky_relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name='hidden_layer_6')
        
        dropout_6 = tf.layers.dropout(hidden_6,rate=0.5,name='dropout_6')
        
        hidden_7 = tf.layers.dense(dropout_6, 100, activation=tf.nn.leaky_relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name='hidden_layer_7')
        dropout_7 = tf.layers.dropout(hidden_7,rate=0.5,name='dropout_7')
        
        hidden_8 = tf.layers.dense(dropout_7, 100, activation=tf.nn.leaky_relu, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  name='hidden_layer_8')
        
        output = tf.layers.dense(hidden_8, 10, name='output_layer')
        
        
    tf.identity(output, name='output')

    y = tf.placeholder(tf.float32, [None, 10], name='label')
    
    return x,y,output
