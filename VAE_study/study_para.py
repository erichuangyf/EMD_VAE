import argparse

# annealing_folders=[]
annealing_folders = ["/global/home/users/yifengh3/VAE/vec_data/vec_model_1"]

parser = argparse.ArgumentParser(description='Plot jets')
parser.add_argument('model_dir')
parser.add_argument('--img_prefix')
parser.add_argument('--utils')


args = parser.parse_args()
print(args)
model_dir = args.model_dir
vae_args_file = model_dir + "/vae_args.dat"


if args.img_prefix:
  file_prefix = args.img_prefix
else:
  file_prefix = model_dir + '/'
print("Saving files with prefix", file_prefix)




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

from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class
import utils.VAE_model_tools_param
from utils.VAE_model_tools_param import build_and_compile_annealing_vae, betaVAEModel, reset_metrics, loss_tracker, myTerminateOnNaN

import pandas

import h5py
import pickle

import pandas

#import h5py
#import pickle
#from scipy.stats import gaussian_kde

from pyjet import cluster
import re
import glob


if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")

def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path

def ptetaphiE_to_Epxpypz(jets):
    pt = jets[:,:,0]
    eta = jets[:,:,1]
    phi = jets[:,:,2]
    E = jets[:,:,3]
    
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    
    newjets = np.zeros(jets.shape)
    newjets[:,:,0] = E
    newjets[:,:,1] = px
    newjets[:,:,2] = py
    newjets[:,:,3] = pz
    
    return newjets

def ptetaphiE_to_ptyphim(jets):
    pt = jets[:,:,0]
    eta = jets[:,:,1]
    phi = jets[:,:,2]
    E = jets[:,:,3]
    
    pz = pt * np.sinh(eta)
    y = 0.5*np.nan_to_num(np.log((E+pz)/(E-pz)))
    
    msqr = np.square(E)-np.square(pt)-np.square(pz)
    msqr[np.abs(msqr) < 1e-6] = 0
    m = np.sqrt(msqr)
    
    newjets = np.zeros(jets.shape)
    newjets[:,:,0] = pt
    newjets[:,:,1] = y
    newjets[:,:,2] = phi
    newjets[:,:,3] = m
    
    return newjets
    
def ptyphim_to_ptetaphiE(jets):
    
    pt = jets[:,:,0]
    y = jets[:,:,1]
    phi = jets[:,:,2]
    m = jets[:,:,3]
    
    eta = np.nan_to_num(np.arcsinh(np.sinh(y)*np.sqrt(1+np.square(m/pt))))
    pz = pt * np.sinh(eta)
    E = np.sqrt(np.square(pz)+np.square(pt)+np.square(m))
    
    newjets = np.zeros(jets.shape)
    newjets[:,:,0] = pt
    newjets[:,:,1] = eta
    newjets[:,:,2] = phi
    newjets[:,:,3] = E
    
    return newjets
    
def center_jets_ptetaphiE(jets):
    cartesian_jets = ptetaphiE_to_Epxpypz(jets)
    sumjet_cartesian = np.sum(cartesian_jets,axis=1)
    
    sumjet_phi = np.arctan2(sumjet_cartesian[:,2],sumjet_cartesian[:,1])
    sumjet_y = 0.5*np.log((sumjet_cartesian[:,0] + sumjet_cartesian[:,-1])/(sumjet_cartesian[:,0] - sumjet_cartesian[:,-1]))
    
    ptyphim_jets = ptetaphiE_to_ptyphim(jets)
    #print(ptyphim_jets[:3,:,:])
    
    transformed_jets = np.copy(ptyphim_jets)
    transformed_jets[:,:,1] = ptyphim_jets[:,:,1] - sumjet_y[:,None]
    transformed_jets[:,:,2] = ptyphim_jets[:,:,2] - sumjet_phi[:,None]
    transformed_jets[:,:,2] = transformed_jets[:,:,2] + np.pi
    transformed_jets[:,:,2] = np.mod(transformed_jets[:,:,2],2*np.pi)
    transformed_jets[:,:,2] = transformed_jets[:,:,2] - np.pi

    transformed_jets[transformed_jets[:,:,0] == 0] = 0
    
    newjets = ptyphim_to_ptetaphiE(transformed_jets)
    return newjets

def kl_loss(z_mean, z_log_var):
    return -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    
    

def get_clustered_pt_eta_phi(pts, locations,R=0.1):
    weights = pts
    outjet = locations
    myjet = np.zeros((weights.shape[-1]),dtype=([('pT', 'f8'), ('eta', 'f8'), ('phi', 'f8'), ('mass', 'f8')]))
    myjet['pT'] = weights
    myjet['eta'] = outjet[:,0]
    myjet['phi'] = outjet[:,1]
    sequence = cluster(myjet,R=R,p=0)
    jets = sequence.inclusive_jets()
    phis = np.array([np.mod(np.pi+jet.phi,2*np.pi)-np.pi for jet in jets])
#     phis = [jet.phi for jet in jets]
    etas = np.array([jet.eta for jet in jets])
    pts = np.array([jet.pt for jet in jets])
    
    return pts, etas, phis


def plot_jets(outs_array, numplot = 3, R=0.02,size=50):
    etalim=5
    #bins=np.linspace(-lim, lim, 126)

    for i in range(numplot):   

        fig, ax = plt.subplots(1, 3,figsize=[15,5],sharey=True)



        outjet = valid_y[i,:,1:]
        weights = valid_y[i,:,0]
        pts, etas, phis = get_clustered_pt_eta_phi(weights, outjet,R=R)
        ax[0].scatter(phis, etas, s = pts*size, alpha = 0.7,linewidths=0)
        ax[0].set_title('Jet'+str(i),y=0.9)

        #ax[0].hist2d(feed_pc[i][:,0],feed_pc[i][:,1],range=[[-lim,lim],[-lim,lim]],bins=bins, norm=LogNorm(0.5, 1000))
        for j in range(2):
            outjet = outs_array[j][0][i,:,1:]
            weights = outs_array[j][0][i,:,0]
            pts, etas, phis = get_clustered_pt_eta_phi(weights, outjet,R=R)
            ax[j+1].scatter(phis, etas, s = pts*size, alpha = 0.7,linewidths=0)
            ax[j+1].set_title('Sample'+ str(j),y=0.9)
            
        for j in range(3):
            ax[j].set_ylabel(r'$\eta$',fontsize=18)
            ax[j].set_xlabel(r'$\phi$',fontsize=18)
            ax[j].set_ylim([-0.7,0.7])
            ax[j].set_xlim([-0.7,0.7])

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        
def plot_KL_logvar(outs_array,xlim=None,ylim=None,showhist=False, numhists=10,hist_ylim=None,hist_xlim=None):
    
    y_pred ,z_mean, z_log_var, _ = outs_array[0]

    KL = kl_loss(z_mean, z_log_var)
    sort_kl = np.flip(np.argsort(np.mean(KL,axis=0)))

    rms_mean = np.sqrt(np.mean(np.square(z_mean),axis=0))

    plt.scatter(np.mean(KL,axis=0),rms_mean,s=5.)

    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
        
    plt.xlabel('KL divergence')
    plt.ylabel(r'$\sqrt{\left\langle \mu^2 \right\rangle}$')
    #plt.show()
    
    if showhist:
#         for i in range(10):
        plt.hist(np.array(KL)[:,sort_kl[:numhists]],bins=np.linspace(0,20,80),stacked=True)
        plt.show()
        if hist_ylim:
            plt.ylim(hist_ylim)
        if hist_xlim:
            plt.xlim(hist_xlim)
    
    return sort_kl



# path to file
fn =  '/global/home/users/yifengh3/VAE/vec_data/B_background.h5'

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

batch_size=150
# save_period=2

# vae.beta.assign(0.001)

# K.set_value(vae.optimizer.lr,1e-4)
# epochs = 1


# history = vae.fit(x=train_x[:10], y=train_y[:10], batch_size=batch_size,
#                 epochs=epochs,verbose=1,#initial_epoch=int(vae.optimizer.iterations/numbatches),
#                 validation_data = (valid_x[:10],valid_y[:10]),
#                 callbacks = None
#               )

print("Preparing to load weights")

def get_epoch(file):
    epoch = int(epoch_string.search(file).group()[1:-1])
    return epoch

def get_beta(file):
    beta = float(beta_string.search(file).group())
    return beta

epoch_string=re.compile('_\d*_')
beta_string=re.compile('\d\.[\w\+-]*')
files = glob.glob(train_output_dir + '/model_weights_end*.hdf5')
for anneal_dir in annealing_folders:
    print("loading annealing folder:{}".format(anneal_dir))
    anneal_output_dir = osp.join(anneal_dir,"end_beta_checkpoint")
    files.extend(glob.glob(anneal_output_dir + '/model_weights_end*.hdf5'))
print("Found files:", files)
files.sort(key=os.path.getmtime)
epochs = np.array([get_epoch(file) for file in files])
betas = np.array([get_beta(file) for file in files])

print(len(files))
# latent_dim =128
# KLs = []
# losses = []
# recons = []
# KLs_array = np.zeros((len(files), latent_dim))
heat_capacity = []
logbeta_min = -6.
logbeta_max = 1.
batch_size = 1000


def generate_batch(data_x, data_y,batch_size,logbeta_min=logbeta_min,logbeta_max=0.):
    idx = 0
    while True:
        batch_x = data_x[idx:idx+batch_size]
        batch_y = data_y[idx:idx+batch_size]
        logbetas = np.random.uniform(low=logbeta_min, high=logbeta_max, size=len(batch_x))
        yield ([batch_x, logbetas],batch_y)
        idx += batch_size
        if idx + batch_size >= len(data_x):
              idx=0


logbetas = np.concatenate((np.logspace(-5.,1.,13),
                          np.logspace(1.,-1.,5)[1:],
                          np.logspace(-1.,1.,5)[1:],
                          np.logspace(1.,-1.,5)[1:],
                          np.logspace(-1.,0.,3)[1:]))

import sys
start=0
for i, file in enumerate(files[start:]):
#     if i%10 == 0:
#         print("Loading file", str(i), "of", str(len(files[start:])))
    sys.stdout.write('\r')
    sys.stdout.write("Loading file {} of {}".format(str(i+1),str(len(files[start:]))))
    sys.stdout.flush()
    vae.load_weights(file)
    ogbetas = np.linspace(3,-7,100).astype(np.float32)
    # print(logbetas)
    # print(np.power(10,logbetas/2)*550)
    result_D = np.zeros(len(logbetas))
    batchsize = 1000
    nsamples = 1
    numbatches = 100
    for i in range(numbatches):
        if i%10 == 0:
            print("Processing batch", i, "of", numbatches)
        x = valid_x[i*batchsize:(i+1)*batchsize]
        y = valid_y[i*batchsize:(i+1)*batchsize]
        result_D += [np.mean(vae.heat_capacity_D([[x,np.ones(len(x))*logbeta],y],nsamples=nsamples)) for logbeta in logbetas]
    result_D = result_D / numbatches
    plt.plot(np.power(10,logbetas),result_D)
    plt.semilogx()
    plt.ylim([0,10])
    plt.show()
    
 