import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
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


from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class
import utils.VAE_model_tools_vm
from utils.VAE_model_tools_vm import build_and_compile_annealing_vae, betaVAEModel, reset_metrics

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

if len(sys.argv) > 1:
  model_dir = sys.argv[1]
  vae_args_file = model_dir + "/vae_args.dat"
else:
  print("No model directory given (first argument)")
  quit()

if len(sys.argv) > 2:
  file_prefix = sys.argv[2]
else:
  file_prefix = model_dir + '/'
print("Saving files with prefix", file_prefix)

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
    
    y_pred ,z_mean, z_log_var, losses, _ = outs_array[0]

    KL=kl_loss(z_mean, z_log_var)
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
fn =  '/scratch/jcollins/monoW-data-parton.h5'

df = pandas.read_hdf(fn,stop=100000)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3)+sum(df.memory_usage(deep=True)) / (1024**3))

data = df.values.reshape((-1,2,4))

HT = np.sum(data[:,:,0],axis=-1)
data[:,:,0] = data[:,:,0]/HT[:,None]
data[:,:,-1] = data[:,:,-1]/HT[:,None]

data = center_jets_ptetaphiE(data)

sig_input = np.zeros((len(data),2,4))
sig_input[:,:,:2] = data[:,:,:2]
sig_input[:,:,2] = np.cos(data[:,:,2])
sig_input[:,:,3] = np.sin(data[:,:,2])
#sig_input[:,:,3] = np.log(data[:,:,3]+1e-8)

data_x = sig_input
data_y = data[:,:,:3]


train_x = data_x[:50000]
train_y = data_y[:50000]
valid_x = data_x[50000:]
valid_y = data_y[50000:]


train_output_dir = model_dir #create_dir(osp.join(output_dir, experiment_name))

with open(vae_args_file,'r') as f:
  vae_arg_dict = json.loads(f.read())

print("\n\n vae_arg_dict:", vae_arg_dict)

vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)

batch_size=100
save_period=2

vae.beta.assign(0.001)

K.set_value(vae.optimizer.lr,1e-4)
epochs = 1


history = vae.fit(x=train_x[:10], y=train_y[:10], batch_size=batch_size,
                epochs=epochs,verbose=1,#initial_epoch=int(vae.optimizer.iterations/numbatches),
                validation_data = (valid_x[:10],valid_y[:10]),
                callbacks = None
              )


def get_epoch(file):
    epoch = int(epoch_string.search(file).group()[1:-1])
    return epoch

def get_beta(file):
    beta = float(beta_string.search(file).group())
    return beta

epoch_string=re.compile('_\d*_')
beta_string=re.compile('\d\.[\w\+-]*')
files = glob.glob(train_output_dir + '/model_weights*.hdf5')
print("Found files:", files)
files.sort(key=os.path.getmtime)
epochs = np.array([get_epoch(file) for file in files])
betas = np.array([get_beta(file) for file in files])

KLs = []
losses = []
recons = []


start=0
for i, file in enumerate(files[start:]):
#     print("Loading", file)
    if i%10 == 0:
        print("Loading file", str(i), "of", str(len(files[start:])))
    vae.load_weights(file)
    vae.beta.assign(betas[i+start])
    outs_array = [vae.predict(valid_x[:1000]) for j in range(1)]
    result = vae.test_step([valid_x[:2000].astype(np.float32),valid_y[:2000].astype(np.float32)])
    
    losses += [result['loss'].numpy()]
    recons += [result['recon_loss'].numpy()]
    KLs += [result['KL loss'].numpy()]

cmap = mpl.cm.get_cmap('viridis')

print(betas)
print(losses)
print(recons)
print(KLs)

ends = np.array([((betas[i+1] > betas[i]) and (betas[i] < betas[i-1])) or ((betas[i+1] < betas[i]) and (betas[i] > betas[i-1])) or (betas[i+1] == betas[i]) for i in range(1,len(betas)-1)])
ends = np.argwhere(ends == True).flatten()+2
ends = np.append(ends,len(betas))
ends = np.insert(ends,0,1)

print(ends)

def split_data(data):
    return[data[ends[i]-1:ends[i+1]] for i in range(len(ends)-1)]

split_betas = split_data(betas)
split_losses = split_data(losses)
split_KLs = split_data(KLs)
split_recons = split_data(recons)

print(split_betas)

n=4
colors = [cmap(1.*i/(n) ) for i in range(n)]
# colors = ['C0','C1','C1','C2','C2']
fig = plt.figure()
for i in range(len(split_betas)):
    style = '-'
    if i%2 == 1:
        style = '--'
    plt.plot(split_betas[i], split_losses[i],linestyle=style,color = colors[int((i+1)/2)])
plt.semilogy()
# plt.ylim([10,None])
plt.semilogx()
plt.xlabel(r'$\beta$')
plt.ylabel(r'Loss')
#plt.xlim(1e-2,1.)
plt.savefig(file_prefix +'loss.png')
plt.show()

fig = plt.figure()
for i in range(len(split_betas)):
    style = '-'
    if i%2 == 1:
        style = '--'
    plt.plot(split_betas[i], split_losses[i]*np.square(split_betas[i]),linestyle=style,color = colors[int((i+1)/2)])
plt.semilogy()
# plt.ylim([10,None])
plt.semilogx()
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\beta^2$ Loss')
#plt.xlim(1e-2,1.)
#plt.ylim(1e-4,None)
plt.savefig(file_prefix +'losstimebetasqr.png')
plt.show()

fig = plt.figure()
for i in range(len(split_betas)):
    style = '-'
    if i%2 == 1:
        style = '--'
    plt.plot(split_betas[i], split_recons[i],linestyle=style,color = colors[int((i+1)/2)])
plt.semilogy()
plt.semilogx()
plt.xlabel(r'$\beta$')
plt.ylabel(r'Recon Loss = EMD$^2$')
#plt.ylim(1e-4,None)
#plt.xlim(1e-2,1.)
plt.savefig(file_prefix +'reconloss.png')
plt.show()

fig = plt.figure()
for i in range(len(split_betas)):
    style = '-'
    if i%2 == 1:
        style = '--'
    plt.plot(split_betas[i], split_KLs[i],linestyle=style,color = colors[int((i+1)/2)])
plt.semilogy()
plt.semilogx()
#plt.xlim(1e-2,1.)
plt.ylim(0.1,None)
plt.xlabel(r'$\beta$')
plt.ylabel(r'KL Loss')
plt.savefig(file_prefix +'KL.png')
plt.show()

fig = plt.figure()
y_pred ,z_mean, z_log_var, losses, _ = outs_array[0]

KL=kl_loss(z_mean, z_log_var)
sort_kl = np.flip(np.argsort(np.mean(KL,axis=0)))

rms_mean = np.sqrt(np.mean(np.square(z_mean),axis=0))

plt.scatter(np.mean(KL,axis=0),rms_mean,s=5.)
plt.xlim([-0.1,None])
plt.ylim([-0.1,None])
plt.xlabel('KL divergence')
plt.ylabel(r'$\sqrt{\left\langle \mu^2 \right\rangle}$')
plt.savefig(file_prefix + 'KL_scatter.png')


plt.show()
