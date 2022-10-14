import tensorflow as tf

# standard numerical library imports
import numpy as np

# energyflow imports
import energyflow as ef
from energyflow.archs import PFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


import pandas
import h5py
import pickle

import datetime
import os


# configs
train, val, test = 0.6, 0.3, 0.1
Phi_sizes, F_sizes = (128, 128, 128), (128, 128, 128)
num_epoch = 200
batch_size = 1000
muon_feature_number = 6

def prepare_data(signal_1, signal_2):
    print("signal_1 data shape: {}".format(signal_1.shape))
    print("signal_2 data shape: {}".format(signal_2.shape))
    # assign labels to signal and background data, 0 for sig1,  1 for sig2
    # (updated since we might get multiple signals) 
    labeled_sig1 = np.append(signal_1.reshape([signal_1.shape[0],-1]),np.zeros((signal_1.shape[0],1)),axis=1)
    labeled_sig2 = np.append(signal_2.reshape([signal_2.shape[0],-1]),np.ones((signal_2.shape[0],1)),axis=1)
    
    # mix two data array into one signal array
    data = np.concatenate((labeled_sig1,labeled_sig2))

    #and shuffle the data
    np.random.shuffle(data)
    
    X = data[:,:-1]
    y = data[:,-1]
    
    print("shape of X: {}".format(X.shape))
    print("shape of Y: {}".format(y.shape))
    
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    total = labeled_sig1.shape[0] + labeled_sig2.shape[0]
    weight_for_0 = (1 / labeled_sig1.shape[0]) * (total / 2.0)
    weight_for_1 = (1 / labeled_sig2.shape[0]) * (total / 2.0)


    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for background: {:.2f}'.format(weight_for_0))
    print('Weight for signal: {:.2f}'.format(weight_for_1))
    
    # To categorical as stipulated in example
    Y = to_categorical(y, num_classes=2)

    # Reshape X to shape (number of jets, 50, 4)
    X = X.reshape(-1,50,3)

    # ignore the pid info
    X = X[:,:,:3]
    
#     # normalizing jets
#     # copied from example
#     import tqdm
#     for x in tqdm.tqdm(X):
#         # now add the status bar :)
#         mask = x[:,0] > 0
#         yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
#         x[mask,1:3] -= yphi_avg
#         x[mask,0] /= x[:,0].sum()
    
    print('Finished preprocessing')
    print("shape of X: {}".format(X.shape))
    print("shape of Y: {}".format(y.shape))
    
    # do train/val/test split 
    (X_train, X_val, X_test,
     Y_train, Y_val, Y_test) = data_split(X, Y, val=val, test=test)
    
    return (X,y), (X_train, X_val, X_test,
     Y_train, Y_val, Y_test), class_weight




def train_pfn(signal_1, signal_2, verbose = 0):
    (X,y), (X_train, X_val, X_test,
     Y_train, Y_val, Y_test), class_weight = prepare_data(signal_1, signal_2)
    print('Model summary:')

    # build architecture
    pfn = PFN(input_dim=X.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, metrics=[tf.keras.metrics.AUC()])

    # now train the model

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.1**(1/4), patience=5, min_lr=1e-5,
                                                    verbose=verbose)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, 
                                                verbose=verbose)

    callbacks = [reduce_lr,early_stop]

    hist1 = pfn.fit(X_train, Y_train,
            epochs=num_epoch,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose)
    
    return pfn, hist1, [(X,y), (X_train, X_val, X_test,
     Y_train, Y_val, Y_test), class_weight]


def analysis(pfn, X_test, Y_test):
    # get predictions on test data
    preds = pfn.predict(X_test, batch_size=10000)

    # get ROC curve
    pfn_fp, pfn_tp, threshs = roc_curve(Y_test[:,1], preds[:,1])

    # get area under the ROC curve
    auc = roc_auc_score(Y_test[:,1], preds[:,1])
    print()
    print('PFN AUC:', auc)
    print()


    # some nicer plot settings 
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True

    # plot the ROC curves
    plt.plot(pfn_tp, 1/pfn_fp, '-', color='black', label='PFN')
    plt.plot(pfn_tp, 1/pfn_tp, '-', color='red', label='random')
    plt.yscale("log")

    # make legend and show plot
    plt.legend(loc='lower left', frameon=False)
    plt.show()

    # some nicer plot settings 
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True

    # plot the ROC curves
    plt.plot(pfn_tp, 1-pfn_fp, '-', color='black', label='PFN')
    plt.plot(pfn_tp, 1-pfn_tp, '-', color='red', label='random')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # make legend and show plot
    plt.legend(loc='lower left', frameon=False)
    plt.show()


def to_combined_data(X):
    return [X[:,:50],X[:,-2:].reshape([-1,6])]

def prepare_data_muon(signal_1, signal_2, signal_1_muon, signal_2_muon):
    # mix two data array into one signal array
    signal_1_combined = np.hstack((signal_1,signal_1_muon))
    signal_2_combined =  np.hstack((signal_2,signal_2_muon))
    
    # assign labels to signal and background data, 0 for sig1,  1 for sig2
    # (updated since we might get multiple signals) 
    labeled_sig1 = np.append(signal_1_combined,np.zeros((signal_1.shape[0],1)),axis=1)
    labeled_sig2 = np.append(signal_2_combined,np.ones((signal_2.shape[0],1)),axis=1)

    data = np.concatenate([labeled_sig1,labeled_sig2])
    
    X = data[:,:-1]
    y = data[:,-1]
    
    print("shape of X: {}".format(X.shape))
    print("shape of Y: {}".format(y.shape))
    
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    total = labeled_sig1.shape[0] + labeled_sig2.shape[0]
    weight_for_0 = (1 / labeled_sig1.shape[0]) * (total / 2.0)
    weight_for_1 = (1 / labeled_sig2.shape[0]) * (total / 2.0)


    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for background: {:.2f}'.format(weight_for_0))
    print('Weight for signal: {:.2f}'.format(weight_for_1))
    
    # To categorical as stipulated in example
    Y = to_categorical(y, num_classes=2)

    # Reshape X to shape (number of jets, 50, 4)
    X = X.reshape(-1,52,3)

    # ignore the pid info
    X = X[:,:,:3]
    
    X = X.astype("float64")
    
    # do train/val/test split 
    (X_train, X_val, X_test,
     Y_train, Y_val, Y_test) = data_split(X, Y, val=val, test=test, shuffle=True)
    
    (X_train, X_val, X_test) = (to_combined_data(X_train),to_combined_data(X_val),to_combined_data(X_test))
    return (X,y), (X_train, X_val, X_test,Y_train, Y_val, Y_test), class_weight


def train_pfn_with_muon(signal_1, signal_2, signal_1_muon, signal_2_muon, verbose=0):
    (X,y), (X_train, X_val, X_test,
     Y_train, Y_val, Y_test), class_weight = prepare_data_muon(signal_1, signal_2, signal_1_muon, signal_2_muon)
    print('Model summary:')

    # build architecture
    pfn = PFN(input_dim=X.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, num_global_features = muon_feature_number, metrics=[tf.keras.metrics.AUC()])

    # now train the model

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.1**(1/4), patience=5, min_lr=1e-5,
                                                    verbose=verbose)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, 
                                                verbose=verbose)

    callbacks = [reduce_lr,early_stop]

    hist1 = pfn.fit(X_train, Y_train,
            epochs=num_epoch,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose)
    
    return pfn, hist1, [(X,y), (X_train, X_val, X_test,
     Y_train, Y_val, Y_test), class_weight]