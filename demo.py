
import argparse
from models import *
from project.datasets import *
from project.utils import *
import cv2
import os
import sys
import math
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

################################################################################

def construct_inceptionv1onfire (x,y, training=False):

    # Build network as per architecture in [Dunnings/Breckon, 2018]

    network = input_data(shape=[None, y, x, 3])

    conv1_7_7 = conv_2d(network, 64, 5, strides=2, activation='relu', name = 'conv1_7_7_s2')

    pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
    pool1_3_3 = local_response_normalization(pool1_3_3)

    conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 128,3, activation='relu', name='conv2_3_3')

    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')

    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
    inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

    # merge the inception_3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

    inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

    #merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

    pool5_7_7 = avg_pool_2d(inception_4a_output, kernel_size=5, strides=1)
    if(training):
        pool5_7_7 = dropout(pool5_7_7, 0.4)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    # if training then add training hyperparameters

    if(training):
        network = regression(loss, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)
    else:
        network = loss;

    model = tflearn.DNN(network, checkpoint_path='inceptiononv1onfire',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model

################################################################################

def construct_inceptionv3onfire(x,y, training=False, enable_batch_norm=True):

    # build network as per architecture

    network = input_data(shape=[None, y, x, 3])

    conv1_3_3 = conv_2d(network, 32, 3, strides=2, activation='relu', name = 'conv1_3_3',padding='valid')
    conv2_3_3 = conv_2d(conv1_3_3, 32, 3, strides=1, activation='relu', name = 'conv2_3_3',padding='valid')
    conv3_3_3 = conv_2d(conv2_3_3, 64, 3, strides=2, activation='relu', name = 'conv3_3_3')

    pool1_3_3 = max_pool_2d(conv3_3_3, 3,strides=2)
    if enable_batch_norm:
        pool1_3_3 = batch_normalization(pool1_3_3)
    conv1_7_7 = conv_2d(pool1_3_3, 80,3, strides=1, activation='relu', name='conv2_7_7_s2',padding='valid')
    conv2_7_7 = conv_2d(conv1_7_7, 96,3, strides=1, activation='relu', name='conv2_7_7_s2',padding='valid')
    pool2_3_3= max_pool_2d(conv2_7_7,3,strides=2)

    inception_3a_1_1 = conv_2d(pool2_3_3,64, filter_size=1, activation='relu', name='inception_3a_1_1')

    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 48, filter_size=1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 64, filter_size=[5,5],  activation='relu',name='inception_3a_3_3')


    inception_3a_5_5_reduce = conv_2d(pool2_3_3, 64, filter_size=1, activation='relu', name = 'inception_3a_5_5_reduce')
    inception_3a_5_5_asym_1 = conv_2d(inception_3a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_3a_5_5_asym_1')
    inception_3a_5_5 = conv_2d(inception_3a_5_5_asym_1, 96, filter_size=[3,3],  name = 'inception_3a_5_5')


    inception_3a_pool = avg_pool_2d(pool2_3_3, kernel_size=3, strides=1,  name='inception_3a_pool')
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

    # merge the inception_3a

    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3, name='inception_3a_output')


    inception_5a_1_1 = conv_2d(inception_3a_output, 96, 1, activation='relu', name='inception_5a_1_1')

    inception_5a_3_3_reduce = conv_2d(inception_3a_output, 64, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3_asym_1 = conv_2d(inception_5a_3_3_reduce, 64, filter_size=[1,7],  activation='relu',name='inception_5a_3_3_asym_1')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_asym_1,96, filter_size=[7,1],  activation='relu',name='inception_5a_3_3')


    inception_5a_5_5_reduce = conv_2d(inception_3a_output, 64, filter_size=1, activation='relu', name = 'inception_5a_5_5_reduce')
    inception_5a_5_5_asym_1 = conv_2d(inception_5a_5_5_reduce, 64, filter_size=[7,1],  name = 'inception_5a_5_5_asym_1')
    inception_5a_5_5_asym_2 = conv_2d(inception_5a_5_5_asym_1, 64, filter_size=[1,7],  name = 'inception_5a_5_5_asym_2')
    inception_5a_5_5_asym_3 = conv_2d(inception_5a_5_5_asym_2, 64, filter_size=[7,1],  name = 'inception_5a_5_5_asym_3')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_asym_3, 96, filter_size=[1,7],  name = 'inception_5a_5_5')


    inception_5a_pool = avg_pool_2d(inception_3a_output, kernel_size=3, strides=1 )
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 96, filter_size=1, activation='relu', name='inception_5a_pool_1_1')

    # merge the inception_5a__
    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], mode='concat', axis=3)



    inception_7a_1_1 = conv_2d(inception_5a_output, 80, 1, activation='relu', name='inception_7a_1_1')
    inception_7a_3_3_reduce = conv_2d(inception_5a_output, 96, filter_size=1, activation='relu', name='inception_7a_3_3_reduce')
    inception_7a_3_3_asym_1 = conv_2d(inception_7a_3_3_reduce, 96, filter_size=[1,3],  activation='relu',name='inception_7a_3_3_asym_1')
    inception_7a_3_3_asym_2 = conv_2d(inception_7a_3_3_reduce, 96, filter_size=[3,1],  activation='relu',name='inception_7a_3_3_asym_2')
    inception_7a_3_3=merge([inception_7a_3_3_asym_1,inception_7a_3_3_asym_2],mode='concat',axis=3)

    inception_7a_5_5_reduce = conv_2d(inception_5a_output, 66, filter_size=1, activation='relu', name = 'inception_7a_5_5_reduce')
    inception_7a_5_5_asym_1 = conv_2d(inception_7a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_7a_5_5_asym_1')
    inception_7a_5_5_asym_2 = conv_2d(inception_7a_3_3_asym_1, 96, filter_size=[1,3],  activation='relu',name='inception_7a_5_5_asym_2')
    inception_7a_5_5_asym_3 = conv_2d(inception_7a_3_3_asym_1, 96, filter_size=[3,1],  activation='relu',name='inception_7a_5_5_asym_3')
    inception_7a_5_5=merge([inception_7a_5_5_asym_2,inception_7a_5_5_asym_3],mode='concat',axis=3)


    inception_7a_pool = avg_pool_2d(inception_5a_output, kernel_size=3, strides=1 )
    inception_7a_pool_1_1 = conv_2d(inception_7a_pool, 96, filter_size=1, activation='relu', name='inception_7a_pool_1_1')

    # merge the inception_7a__
    inception_7a_output = merge([inception_7a_1_1, inception_7a_3_3, inception_7a_5_5, inception_7a_pool_1_1], mode='concat', axis=3)



    pool5_7_7=global_avg_pool(inception_7a_output)
    if(training):
        pool5_7_7=dropout(pool5_7_7,0.4)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    if(training):
        network = regression(loss, optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    else:
        network=loss

    model = tflearn.DNN(network, checkpoint_path='inceptionv3',
                        max_checkpoints=1, tensorboard_verbose=0)

    return model

################################################################################

# InceptionV4 : definition of inception_block_a

def inception_block_a(input_a):

    inception_a_conv1_1_1 = conv_2d(input_a,96,1,activation='relu',name='inception_a_conv1_1_1')

    inception_a_conv1_3_3_reduce = conv_2d(input_a,64,1,activation='relu',name='inception_a_conv1_3_3_reduce')
    inception_a_conv1_3_3 = conv_2d(inception_a_conv1_3_3_reduce,96,3,activation='relu',name='inception_a_conv1_3_3')

    inception_a_conv2_3_3_reduce = conv_2d(input_a,64,1,activation='relu',name='inception_a_conv2_3_3_reduce')
    inception_a_conv2_3_3_sym_1 = conv_2d(inception_a_conv2_3_3_reduce,96,3,activation='relu',name='inception_a_conv2_3_3')
    inception_a_conv2_3_3 = conv_2d(inception_a_conv2_3_3_sym_1,96,3,activation='relu',name='inception_a_conv2_3_3')

    inception_a_pool = avg_pool_2d(input_a,kernel_size=3,name='inception_a_pool',strides=1)
    inception_a_pool_1_1 = conv_2d(inception_a_pool,96,1,activation='relu',name='inception_a_pool_1_1')

    # merge inception_a

    inception_a = merge([inception_a_conv1_1_1,inception_a_conv1_3_3,inception_a_conv2_3_3,inception_a_pool_1_1],mode='concat',axis=3)

    return inception_a


################################################################################

# InceptionV4 : definition of reduction_block_a

def reduction_block_a(reduction_input_a):

    reduction_a_conv1_1_1 = conv_2d(reduction_input_a,384,3,strides=2,padding='valid',activation='relu',name='reduction_a_conv1_1_1')

    reduction_a_conv2_1_1 = conv_2d(reduction_input_a,192,1,activation='relu',name='reduction_a_conv2_1_1')
    reduction_a_conv2_3_3 = conv_2d(reduction_a_conv2_1_1,224,3,activation='relu',name='reduction_a_conv2_3_3')
    reduction_a_conv2_3_3_s2 = conv_2d(reduction_a_conv2_3_3,256,3,strides=2,padding='valid',activation='relu',name='reduction_a_conv2_3_3_s2')

    reduction_a_pool = max_pool_2d(reduction_input_a,strides=2,padding='valid',kernel_size=3,name='reduction_a_pool')

    # merge reduction_a

    reduction_a = merge([reduction_a_conv1_1_1,reduction_a_conv2_3_3_s2,reduction_a_pool],mode='concat',axis=3)

    return reduction_a

################################################################################

# InceptionV4 : definition of inception_block_b

def inception_block_b(input_b):

    inception_b_1_1 = conv_2d(input_b, 384, 1, activation='relu', name='inception_b_1_1')

    inception_b_3_3_reduce = conv_2d(input_b, 192, filter_size=1, activation='relu', name='inception_b_3_3_reduce')
    inception_b_3_3_asym_1 = conv_2d(inception_b_3_3_reduce, 224, filter_size=[1,7],  activation='relu',name='inception_b_3_3_asym_1')
    inception_b_3_3 = conv_2d(inception_b_3_3_asym_1, 256, filter_size=[7,1],  activation='relu',name='inception_b_3_3')


    inception_b_5_5_reduce = conv_2d(input_b, 192, filter_size=1, activation='relu', name = 'inception_b_5_5_reduce')
    inception_b_5_5_asym_1 = conv_2d(inception_b_5_5_reduce, 192, filter_size=[7,1],  name = 'inception_b_5_5_asym_1')
    inception_b_5_5_asym_2 = conv_2d(inception_b_5_5_asym_1, 224, filter_size=[1,7],  name = 'inception_b_5_5_asym_2')
    inception_b_5_5_asym_3 = conv_2d(inception_b_5_5_asym_2, 224, filter_size=[7,1],  name = 'inception_b_5_5_asym_3')
    inception_b_5_5 = conv_2d(inception_b_5_5_asym_3, 256, filter_size=[1,7],  name = 'inception_b_5_5')


    inception_b_pool = avg_pool_2d(input_b, kernel_size=3, strides=1 )
    inception_b_pool_1_1 = conv_2d(inception_b_pool, 128, filter_size=1, activation='relu', name='inception_b_pool_1_1')

    # merge the inception_b

    inception_b_output = merge([inception_b_1_1, inception_b_3_3, inception_b_5_5, inception_b_pool_1_1], mode='concat', axis=3)

    return inception_b_output

################################################################################

# InceptionV4 : definition of reduction_block_b

def reduction_block_b(reduction_input_b):

    reduction_b_1_1 = conv_2d(reduction_input_b,192,1,activation='relu',name='reduction_b_1_1')
    reduction_b_1_3 = conv_2d(reduction_b_1_1,192,3,strides=2,padding='valid',name='reduction_b_1_3')

    reduction_b_3_3_reduce = conv_2d(reduction_input_b, 256, filter_size=1, activation='relu', name='reduction_b_3_3_reduce')
    reduction_b_3_3_asym_1 = conv_2d(reduction_b_3_3_reduce, 256, filter_size=[1,7],  activation='relu',name='reduction_b_3_3_asym_1')
    reduction_b_3_3_asym_2 = conv_2d(reduction_b_3_3_asym_1, 320, filter_size=[7,1],  activation='relu',name='reduction_b_3_3_asym_2')
    reduction_b_3_3=conv_2d(reduction_b_3_3_asym_2,320,3,strides=2,activation='relu',padding='valid',name='reduction_b_3_3')

    reduction_b_pool = max_pool_2d(reduction_input_b,kernel_size=3,strides=2,padding='valid')

    # merge the reduction_b

    reduction_b_output = merge([reduction_b_1_3,reduction_b_3_3,reduction_b_pool],mode='concat',axis=3)

    return reduction_b_output

################################################################################

# InceptionV4 : defintion of inception_block_c

def inception_block_c(input_c):
    inception_c_1_1 = conv_2d(input_c, 256, 1, activation='relu', name='inception_c_1_1')
    inception_c_3_3_reduce = conv_2d(input_c, 384, filter_size=1, activation='relu', name='inception_c_3_3_reduce')
    inception_c_3_3_asym_1 = conv_2d(inception_c_3_3_reduce, 256, filter_size=[1,3],  activation='relu',name='inception_c_3_3_asym_1')
    inception_c_3_3_asym_2 = conv_2d(inception_c_3_3_reduce, 256, filter_size=[3,1],  activation='relu',name='inception_c_3_3_asym_2')
    inception_c_3_3=merge([inception_c_3_3_asym_1,inception_c_3_3_asym_2],mode='concat',axis=3)

    inception_c_5_5_reduce = conv_2d(input_c, 384, filter_size=1, activation='relu', name = 'inception_c_5_5_reduce')
    inception_c_5_5_asym_1 = conv_2d(inception_c_5_5_reduce, 448, filter_size=[1,3],  name = 'inception_c_5_5_asym_1')
    inception_c_5_5_asym_2 = conv_2d(inception_c_5_5_asym_1, 512, filter_size=[3,1],  activation='relu',name='inception_c_5_5_asym_2')
    inception_c_5_5_asym_3 = conv_2d(inception_c_5_5_asym_2, 256, filter_size=[1,3],  activation='relu',name='inception_c_5_5_asym_3')

    inception_c_5_5_asym_4 = conv_2d(inception_c_5_5_asym_2, 256, filter_size=[3,1],  activation='relu',name='inception_c_5_5_asym_4')
    inception_c_5_5=merge([inception_c_5_5_asym_4,inception_c_5_5_asym_3],mode='concat',axis=3)


    inception_c_pool = avg_pool_2d(input_c, kernel_size=3, strides=1 )
    inception_c_pool_1_1 = conv_2d(inception_c_pool, 256, filter_size=1, activation='relu', name='inception_c_pool_1_1')

    # merge the inception_c

    inception_c_output = merge([inception_c_1_1, inception_c_3_3, inception_c_5_5, inception_c_pool_1_1], mode='concat', axis=3)

    return inception_c_output

################################################################################

def construct_inceptionv4onfire(x,y, training=False, enable_batch_norm=True):

    network = input_data(shape=[None, y, x, 3])

    #stem of inceptionV4

    conv1_3_3 = conv_2d(network,32,3,strides=2,activation='relu',name='conv1_3_3_s2',padding='valid')
    conv2_3_3 = conv_2d(conv1_3_3,32,3,activation='relu',name='conv2_3_3')
    conv3_3_3 = conv_2d(conv2_3_3,64,3,activation='relu',name='conv3_3_3')
    b_conv_1_pool = max_pool_2d(conv3_3_3,kernel_size=3,strides=2,padding='valid',name='b_conv_1_pool')
    if enable_batch_norm:
        b_conv_1_pool = batch_normalization(b_conv_1_pool)
    b_conv_1_conv = conv_2d(conv3_3_3,96,3,strides=2,padding='valid',activation='relu',name='b_conv_1_conv')
    b_conv_1 = merge([b_conv_1_conv,b_conv_1_pool],mode='concat',axis=3)

    b_conv4_1_1 = conv_2d(b_conv_1,64,1,activation='relu',name='conv4_3_3')
    b_conv4_3_3 = conv_2d(b_conv4_1_1,96,3,padding='valid',activation='relu',name='conv5_3_3')

    b_conv4_1_1_reduce = conv_2d(b_conv_1,64,1,activation='relu',name='b_conv4_1_1_reduce')
    b_conv4_1_7 = conv_2d(b_conv4_1_1_reduce,64,[1,7],activation='relu',name='b_conv4_1_7')
    b_conv4_7_1 = conv_2d(b_conv4_1_7,64,[7,1],activation='relu',name='b_conv4_7_1')
    b_conv4_3_3_v = conv_2d(b_conv4_7_1,96,3,padding='valid',name='b_conv4_3_3_v')
    b_conv_4 = merge([b_conv4_3_3_v, b_conv4_3_3],mode='concat',axis=3)

    b_conv5_3_3 = conv_2d(b_conv_4,192,3,padding='valid',activation='relu',name='b_conv5_3_3',strides=2)
    b_pool5_3_3 = max_pool_2d(b_conv_4,kernel_size=3,padding='valid',strides=2,name='b_pool5_3_3')
    if enable_batch_norm:
        b_pool5_3_3 = batch_normalization(b_pool5_3_3)
    b_conv_5 = merge([b_conv5_3_3,b_pool5_3_3],mode='concat',axis=3)
    net = b_conv_5

    # inceptionV4 modules

    net=inception_block_a(net)

    net=inception_block_b(net)

    net=inception_block_c(net)

    pool5_7_7=global_avg_pool(net)
    if(training):
        pool5_7_7=dropout(pool5_7_7,0.4)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    if(training):
        network = regression(loss, optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    else:
        network=loss

    model = tflearn.DNN(network, checkpoint_path='inceptionv4onfire',
                        max_checkpoints=1, tensorboard_verbose=0)

    return model


def boxhandler(bbox):
    for box in bbox:
        box[0] = int(box[0]-box[2]/2)
        box[1] = int(box[1]-box[3]/2)
        box[2] = int(box[0]+box[2])
        box[3] = int(box[1]+box[3])
    return bbox


def boxcover(image,box):
    w=image.shape[1]
    h=image.shape[0]
    box[1] = max(0, box[1]-5)
    box[3] = min(h, box[3]+5)
    box[0] = max(0, box[0]-5)
    box[2] = min(w, box[2]+5)
    image[box[1]:box[3],box[0]:box[2],0] = 0
    image[box[1]:box[3],box[0]:box[2],1] = 0
    image[box[1]:box[3],box[0]:box[2],2] = 0
    return image

def detect():
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

def imagedetect():
    model = construct_inceptionv4onfire(224, 224, training=False)
    model.load(os.path.join("models/InceptionV4-OnFire", "inceptionv4onfire"), weights_only=False)
    print("Loaded CNN network weights ...")
    rows = 224
    cols = 224
    if '.txt' in Source:
        f = open(Source, 'r')
        imagepaths = f.readlines()
        f.close()
    else:
        imagepaths = os.listdir(Source)
        for i in range(len(imagepaths)):
            imagepaths[i] = Source+'/'+imagepaths[i]
    imagepaths.sort()
    result = []
    if savetxt:
        F = open('result.txt', 'w')
    for imagepath in imagepaths:
        if '.jpg' in imagepath or '.JPG' in imagepath:
            imagepath = imagepath.strip('\n')
            image = cv2.imread(imagepath)
            print(imagepath)
            width = image.shape[1]
            height = image.shape[0]
            if FLremove:
                txtpath = imagepath.strip('.jpg')+'.txt'
                if os.path.exists(txtpath):
                    f = open(txtpath, 'r')
                    lines = f.readlines()
                    f.close()
                    bbox = []
                    for line in lines:
                        line = line.strip('\n').split(' ')
                        if line[0] == '1':
                            box = [float(line[1]) * width, float(line[2]) * height, float(line[3]) * width, float(line[4]) * height]
                            bbox.append(box)
                    if len(bbox) > 0:
                        boxhandler(bbox)
                        for box in bbox:
                            image = boxcover(image, box)
            small_image = cv2.resize(image, (rows, cols), cv2.INTER_AREA)
            output = model.predict([small_image])
            if savetxt:
                F.write(imagepath.strip('\n') + ' ' + str(output[0][0]) + '\n')
            if saveimg:
                if round(output[0][0]) > 0.5:  # equiv. to 0.5 threshold in [Dunnings / Breckon, 2018],  [Samarth/Bhowmik/Breckon, 2019] test code
                    cv2.rectangle(image, (0, 0), (width, height), (0, 0, 255), 50)
                    cv2.putText(image, 'FIRE', (int(width / 16), int(height / 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10, cv2.LINE_AA);
                else:
                    cv2.rectangle(image, (0, 0), (width, height), (0, 255, 0), 50)
                    cv2.putText(image, 'CLEAR', (int(width / 16), int(height / 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10, cv2.LINE_AA);
                cv2.imwrite(Output+'/'+imagepath.strip('\n').split('/')[len(imagepath.split('/'))-1], image)
                
    if savetxt:
        F.close()
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/fires.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/fire_lamp.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.11, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--Source', type=str, default='output', help='source')
    parser.add_argument('--savetxt', action='store_true',help='save output')
    parser.add_argument('--saveimg', action='store_true',help='save img')
    parser.add_argument('--FLremove',action='store_true',help='remove firelike')
    parser.add_argument('--imgnum', type=int, default=-1)
    parser.add_argument('--Output', type=str, default='output')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    Source, savetxt, saveimg, FLremove, imgnum, Output = opt.Source, opt.savetxt, opt.saveimg, opt.FLremove, opt.imgnum, opt.Output
    #print(savetxt,saveimg,FLremove)
    with torch.no_grad():
        detect()
    imagedetect()
