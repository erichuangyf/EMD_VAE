#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[2]:


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import tensorflow.keras as keras
import tensorflow.keras.backend as K

import os
import os.path as osp
import sys
import json

import numpy as np
#from scipy import linalg as LA

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# from colorspacious import cspace_converter
from collections import OrderedDict


import tensorflow.keras as keras
import tensorflow.keras.backend as K

import numpy as np
import pandas


# In[3]:


import utils.VAE_model_tools_param
from utils.VAE_model_tools_param import build_and_compile_annealing_vae, betaVAEModel, reset_metrics, loss_tracker, myTerminateOnNaN


# In[4]:


fn =  '/global/home/users/yifengh3/VAE/vec_data/B_background.h5'
vae_args_file = "/global/home/users/yifengh3/VAE/vec_data/vec_model_no_anneal/vae_args.dat"
model_dir = "/global/home/users/yifengh3/VAE/vec_data/vec_model_no_anneal"
numbatches = 1000

df = pandas.read_hdf(fn,stop=1000000)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3)+sum(df.memory_usage(deep=True)) / (1024**3))

# Data file contains, for each event, 50 particles (with zero padding), each particle with pT, eta, phi, E.
# we only need first 200th entries, weight is not required here
numparts=50
data = df.values[:,:200]
print(data.shape)
data = data.reshape((-1,numparts,4))
data = data.astype(float)

# Normalize pTs so that HT = 1
HT = np.sum(data[:,:,0],axis=-1)
data[:,:,0] = data[:,:,0]/HT[:,None]
data[:,:,-1] = data[:,:,-1]/HT[:,None]

# Inputs x to NN will be: pT, eta, cos(phi), sin(phi), log E
# Separated phi into cos and sin for continuity around full detector, so make things easier for NN.
# Also adding the log E is mainly because it seems like it should make things easier for NN, since there is an exponential spread in particle energies.
# Feel free to change these choices as desired. E.g. px, py might be equally as good as pt, sin, cos.
sig_input = np.zeros((len(data),numparts,4))
sig_input[:,:,:2] = data[:,:,:2]
sig_input[:,:,2] = np.cos(data[:,:,2])
sig_input[:,:,3] = np.sin(data[:,:,2])
#sig_input[:,:,4] = np.log(data[:,:,3]+1e-8)


data_x = sig_input
# Event 'labels' y are [pT, eta, phi], which is used to calculate EMD to output which is also pT, eta, phi.
data_y = data[:,:,:3]

numtrain = 500000
train_x = data_x[:numtrain]
train_y = data_y[:numtrain]
valid_x = data_x[numtrain:numtrain+100000]
valid_y = data_y[numtrain:numtrain+100000]

# train_output_dir = osp.join(model_dir,"end_beta_checkpoint") #create_dir(osp.join(output_dir, experiment_name))
train_output_dir = model_dir


with open(vae_args_file,'r') as f:
     vae_arg_dict = json.loads(f.read())

print("\n\n vae_arg_dict:", vae_arg_dict)

vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)


# In[5]:


logbetas = np.linspace(1,-6,50).astype(np.float32)


# In[6]:


model_dir = "/global/home/users/yifengh3/VAE/vec_data/vec_model_no_anneal"
model_name = "model_weights_end_1429_betar_1.0e+00.hdf5"


# In[7]:


vae.load_weights(os.path.join(model_dir,model_name))
print("model loaded with {}".format(os.path.join(model_dir,model_name)))


# In[8]:


result_D = np.zeros(len(logbetas))
batchsize = 1000
nsamples = 1
# numbatches = 5
for i in tqdm(range(numbatches)):
#     if i%10 == 0:
#         print("Processing batch", i, "of", numbatches)
    i = i %(valid_x.shape[0]//batchsize)
    x = valid_x[i*batchsize:(i+1)*batchsize]
    y = valid_y[i*batchsize:(i+1)*batchsize]
    result_D += np.array([np.mean(vae.heat_capacity_D([[x,np.ones(len(x))*logbeta],y],nsamples=nsamples)) for logbeta in logbetas])/ numbatches


# In[12]:


plt.figure(figsize=(10,10))
plt.plot(np.power(10,logbetas),result_D)
plt.semilogx()
plt.savefig(os.path.join(model_dir, "result_d.png"))
# plt.ylim([0,100])
plt.show()


# In[ ]:


result_K = np.zeros(len(logbetas))
batchsize = 1000
nsamples = 1
# numbatches = 5
for i in tqdm(range(numbatches)):
    i = i %(valid_x.shape[0]//batchsize)
    x = valid_x[i*batchsize:(i+1)*batchsize]
    y = valid_y[i*batchsize:(i+1)*batchsize]
    result_K += np.array([np.mean(vae.heat_capacity_KL([[x,np.ones(len(x))*logbeta],y])) for logbeta in logbetas])/ numbatches


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(np.power(10,logbetas),result_K)
plt.semilogx()
plt.savefig(os.path.join(model_dir, "result_k.png"))
plt.show()


# In[ ]:


# np.savez("result_non_anneal.npz", logbetas = logbetas, result_K=result_K, result_D=result_D)


# In[ ]:


recon_loss = np.zeros(len(logbetas))
batchsize = 1000
nsamples = 1
# numbatches = 5
for i in tqdm(range(numbatches)):
    i = i %(valid_x.shape[0]//batchsize)
    x = valid_x[i*batchsize:(i+1)*batchsize]
    y = valid_y[i*batchsize:(i+1)*batchsize]
    recon_loss += np.array([np.mean(vae.recon_loss(y.astype("float32"), vae([x,np.ones(len(x))*logbeta], training=False)[0])) for logbeta in logbetas]) / numbatches


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(np.power(10,logbetas),recon_loss)
plt.semilogx()
plt.savefig(os.path.join(model_dir, "recon_loss.png"))
plt.show()


# In[ ]:


kl_loss = np.zeros(len(logbetas))
batchsize = 1000
nsamples = 1
# numbatches = 5
for i in tqdm(range(numbatches)):
    i = i %(valid_x.shape[0]//batchsize)
    x = valid_x[i*batchsize:(i+1)*batchsize]
    y = valid_y[i*batchsize:(i+1)*batchsize]
    kl_loss += np.array([np.mean(vae.KL_loss(*vae([x,np.ones(len(x))*logbeta], training=False)[1:3])) for logbeta in logbetas]) / numbatches


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(np.power(10,logbetas),kl_loss)
plt.semilogx()
plt.savefig(os.path.join(model_dir, "kl_loss.png"))
plt.show()

individule_kl_loss = np.zeros([len(logbetas),256])
batchsize = 1000
nsamples = 1
# numbatches = 5
def kl_loss_func(z_mean, z_log_var):
    return -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
for i in tqdm(range(numbatches)):
    i = i %(valid_x.shape[0]//batchsize)
    x = valid_x[i*batchsize:(i+1)*batchsize]
    y = valid_y[i*batchsize:(i+1)*batchsize]
    individule_kl_loss +=  np.array(
        [np.mean(
            kl_loss_func(
                *vae([x,np.ones(len(x))*logbeta], training=False)[1:3]), 
            axis=0) for logbeta in logbetas]) / numbatches


for kl_loss in individule_kl_loss.T:
    plt.plot(np.power(10,logbetas), kl_loss)
plt.semilogx()
plt.savefig(os.path.join(model_dir, "latent space kl loss.png"))
plt.show()


# In[ ]:


np.savez("result_non_anneal.npz",
         logbetas = logbetas, 
         result_K=result_K, 
         result_D=result_D, 
         recon_loss=recon_loss, 
         kl_loss= kl_loss,
        individule_kl_loss=individule_kl_loss)
