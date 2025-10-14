
"""
Created on March 17 20:26:42 2023
Tensorflow & Keras version: 2.7.0
@author: kg
"""
import os
import keras
from tensorflow.python.ops import gen_nn_ops

import tensorflow as tf 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Activation, Concatenate, Reshape

import numpy as np
import scipy.io as sio
import gc
import mat73

def test_model(input1):

    output1 = input1

    output1 = Conv2D(6, (1, 32), strides=(1,16),activation='relu', padding='same')(output1)
    output1 = Conv2D(6, (2, 1), strides=(2,1),activation='relu', padding='same')(output1)
    output1 = Conv2D(6, (2, 1), strides=(2,1),activation='relu', padding='same')(output1)
    
    output1 = Flatten()(output1)
    
    output1 = Dropout(0.4)(output1)
    output1 = Dense(512, activation='relu')(output1)
    output1 = Dropout(0.4)(output1)  
    output1 = Dense(128, activation='relu')(output1)
    output1 = Dropout(0.4)(output1)    
    
    output1 = Dense(2, activation='softmax')(output1)

    return output1

#######################################################################################################################
############ Layer-Wise Relevance Propagation (LRP): Calculation of relevance score at input data point ###############
def backprop_conv2d_1(w, b, a, r, strides=(1, 1, 16, 1)):
  hb = a*0 + K.max(a)
  lb = a*0 + K.min(a)
  w_p = K.maximum(w, 0.)
  b_p = K.maximum(b, 0.)
  w_n = K.minimum(w, 0.)
  b_n = K.minimum(b, 0.)

  z_p = K.conv2d(a, kernel=w_p, strides=strides[1:-1], padding='same') + b_p
  z_p -= K.conv2d(lb, kernel=w_p, strides=strides[1:-1], padding='same') + b_p
  z_p -= K.conv2d(hb, kernel=w_n, strides=strides[1:-1], padding='same') + b_n
  s = r / z_p
  c = tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_p, s, strides, padding='SAME')
  cp = tf.compat.v1.nn.conv2d_backprop_input(K.shape(hb), w_n, s, strides, padding='SAME')
  cm = tf.compat.v1.nn.conv2d_backprop_input(K.shape(lb), w_p, s, strides, padding='SAME')
  
  return a*c + lb*cp + hb*cm

    
def backprop_conv2d_2(w, b, a, r, strides=(1, 2, 1, 1)):
  alpha=1
  epsilon=1e-7
  beta = 1 - alpha
  w_p = K.maximum(w, 0.)
  b_p = K.maximum(b, 0.)
  z_p = K.conv2d(a, kernel=w_p, strides=strides[1:-1], padding='same') + b_p + epsilon
  s_p = r / z_p
  c_p = tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_p, s_p, strides, padding='SAME')
  
  w_n = K.minimum(w, 0.)
  b_n = K.minimum(b, 0.)
  z_n = K.conv2d(a, kernel=w_n, strides=strides[1:-1], padding='same') + b_n - epsilon
  s_n = r / z_n
  c_n = tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_n, s_n, strides, padding='SAME')

  return a * (alpha * c_p + beta * c_n)

def backprop_conv2d_3(w, b, a, r, strides=(1, 2, 1, 1)):
  alpha=1
  epsilon=1e-7
  beta = 1 - alpha
  w_p = K.maximum(w, 0.)
  b_p = K.maximum(b, 0.)
  z_p = K.conv2d(a, kernel=w_p, strides=strides[1:-1], padding='same') + b_p + epsilon
  s_p = r / z_p
  c_p = tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_p, s_p, strides, padding='SAME')

  w_n = K.minimum(w, 0.)
  b_n = K.minimum(b, 0.)
  z_n = K.conv2d(a, kernel=w_n, strides=strides[1:-1], padding='same') + b_n - epsilon
  s_n = r / z_n
  c_n = tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_n, s_n, strides, padding='SAME')

  return a * (alpha * c_p + beta * c_n)

def backprop_flatten(a, r):
  shape = a.get_shape().as_list()
  shape[0] = -1
  return K.reshape(r, shape) 

def backprop_fc(w, b, a, r):
  alpha=1
  epsilon=1e-7
  beta = 1 - alpha
  w_p = K.maximum(w, 0.)
  b_p = K.maximum(b, 0.)
  z_p = K.dot(a, w_p) + b_p + epsilon
  s_p = r / z_p
  c_p = K.dot(s_p, K.transpose(w_p))
  
  w_n = K.minimum(w, 0.)
  b_n = K.minimum(b, 0.)
  z_n = K.dot(a, w_n) + b_n - epsilon
  s_n = r / z_n
  c_n = K.dot(s_n, K.transpose(w_n))

  return a * (alpha * c_p + beta * c_n)

def compute_relevances(output, names, weights, activations):
  r = output
  num_layers = len(names)
  for i in range(num_layers-2, -1, -1):
    if 'dense' in names[i + 1]:
      r = backprop_fc(weights[i + 1][0], weights[i + 1][1], activations[i], r)
    elif 'flatten' in names[i + 1]:
      r = backprop_flatten(activations[i], r)
    elif names[i + 1]=='conv2d':
      r = backprop_conv2d_1(weights[i + 1][0], weights[i + 1][1], activations[i], r)
    elif names[i + 1]=='conv2d_1':
      r = backprop_conv2d_2(weights[i + 1][0], weights[i + 1][1], activations[i], r)
    elif names[i + 1]=='conv2d_2':
      r = backprop_conv2d_3(weights[i + 1][0], weights[i + 1][1], activations[i], r)
    else:
      r = r
  return r
#######################################################################################################################
#######################################################################################################################

Dir = 'D:\\fNIRS_MDD_DL/ExplainableDL' 

acc_final = np.zeros(3)

import time

i=0
savename = 'Results_CAD_.mat'
acc = {}

#Data load
temp_data = mat73.loadmat(Dir + '/epopy_vft.mat') #10(fold) List, Preprocessing and Segmentation within 10 Fold was performed with Matlab  
all_numID = temp_data['epopy']['NumID'] #ID of subjects in each fold 
all_trainX = temp_data['epopy']['trainX'] #Size: (The number of Samples, The number of channel * 2 (HbO and HbR), The number of Subjects)
all_trainY = temp_data['epopy']['trainY'] 
all_testX = temp_data['epopy']['testX'] #Size: (The number of Samples, The number of channel * 2 (HbO and HbR), The number of Subjects)
all_testY = temp_data['epopy']['testY'] 

#Variable declaration  
nFold=10 #The number of folds
nCh=68 #The number of channels
nClass=2 #The number of class
nEpoch=30 #The number of training
nSub=116 #The number of subjects 
nSamples=all_trainX[0].shape[0] #The number of data point within temporal axis
nSubinTestFold=all_testX[0].shape[2] #The number of subject within test dataset

test_acc = np.zeros((nFold,nEpoch))
test_loss = np.zeros((nFold,nEpoch))
train_acc = np.zeros((nFold,nEpoch))
train_loss = np.zeros((nFold,nEpoch))
accuracy = np.zeros((nFold))
r_final = np.zeros([nFold,nSubinTestFold,nCh,nSamples,2]) #Last term: the number of hemodynamic responses (HbO and HbR) 
sensitivity = np.zeros(10)
specificity = np.zeros(10)

seed = 0;
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(seed)


for k in range(nFold):
    gc.collect()
    K.clear_session()
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    K.set_session(sess) 
    tf.random.set_seed(seed)

    datatype = 'float32'
    
    tempTrain_x=all_trainX[k]
    tempTrain_y=all_trainY[k]
    tempTest_x=all_testX[k]
    tempTest_y=all_testY[k]
    temp_id=all_numID[k]
    
    #Input data construction
    train_oxy=tempTrain_x[:,:nCh,:].reshape(nSamples, nCh, -1, 1).transpose([2, 1, 0, 3])
    train_deoxy=tempTrain_x[:,nCh:,:].reshape(nSamples, nCh, -1, 1).transpose([2, 1, 0, 3])
    test_oxy=tempTest_x[:,:nCh,:].reshape(nSamples, nCh, -1, 1).transpose([2, 1, 0, 3])
    test_deoxy=tempTest_x[:,nCh:,:].reshape(nSamples, nCh, -1, 1).transpose([2, 1, 0, 3])

    train_x = np.concatenate([train_oxy, train_deoxy], axis=3)
    train_y = tempTrain_y.reshape(nClass, -1).transpose()
    
    test_x = np.concatenate([test_oxy, test_deoxy], axis=3)
    test_y = tempTest_y.reshape(nClass, -1).transpose()
    
    #Training of model
    Input_data = Input(shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3]))
    output_data = test_model(Input_data)
    
    model = Model(Input_data, output_data)

    opt = optimizers.Adam(learning_rate=0.01, decay=0.3)
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    
    print('fold: ', k)
    history1 = model.fit(train_x, train_y,
                         epochs=nEpoch, batch_size=16, 
                         shuffle=True, verbose=1, validation_data=(test_x,test_y))
    
    #Performance of model
    results = model.evaluate(test_x,test_y)
    pred = model.predict(test_x)
    
    pred_class = np.round(pred)
    result_class = np.sum(pred_class*test_y, axis=1)
    hc = np.sum(test_y, axis=0)[0].astype(int)
    mdd = np.sum(test_y, axis=0)[1].astype(int)
    
    sensitivity[k] = np.sum(result_class[hc:])/mdd
    specificity[k] = np.sum(result_class[:hc])/hc

    #Implementation of LRP
    names, activations, weights = [], [], []
    for layer in model.layers:
      name = layer.name if layer.name != 'activation3' else 'fc_out'
      names.append(name)
      activations.append(layer.output)
      weights.append(layer.get_weights())
      
    r = compute_relevances(pred, names, weights, activations)
    lrp_runner = K.function(inputs=[model.input, ], outputs=[r, ])
    r_final[k,:(hc+mdd),:,:,:] = lrp_runner([test_x,])[0]
    
    #Model history
    print('loss: ', results[0], '\n', 'acc: ', results[1])
    accuracy[k] = results[1]
    test_acc[k,] = history1.history['val_accuracy']
    test_loss[k,] = history1.history['val_loss']
    train_acc[k,] = history1.history['accuracy']
    train_loss[k,] = history1.history['loss']

    acc_final = np.mean(accuracy[:k+1])
    print(nEpoch, " Epoch: ",)
    print(acc_final)
    
    #Extraction of result
    acc['test_acc'] = test_acc[:,:]
    acc['test_loss'] = test_loss[:,:]
    acc['train_acc'] = train_acc[:,:]
    acc['train_loss'] = train_loss[:,:]
    acc['acc'] = accuracy
    acc['relevance'] = r_final
    acc['sen'] = sensitivity
    acc['spe'] = specificity

# Save result
sio.savemat(Dir + '/' + savename, acc)   