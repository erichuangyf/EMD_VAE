import argparse
import os
import os.path as osp
import sys
import json
import numpy as np

parser = argparse.ArgumentParser(description='Plot VAE training info')
parser.add_argument('model_dir')
parser.add_argument('--img_prefix')
parser.add_argument('--utils')
parser.add_argument('--img_title')
parser.add_argument('--center',action="store_true")
parser.add_argument('--boost',action="store_true")
parser.add_argument('--parton',action="store_true")
parser.add_argument('--tj',action="store_true")
parser.add_argument('--data_path',default='/sdf/group/ml/EMDVAE_representation_learning/')
parser.add_argument('--time_order',action="store_true")
parser.add_argument('--top',action="store_true")

args = parser.parse_args()
print(args)
model_dir = args.model_dir
vae_args_file = model_dir + "/vae_args.dat"

with open(vae_args_file,'r') as f:
  vae_arg_dict = json.loads(f.read())

print("\n\n vae_arg_dict:", vae_arg_dict)

latent_dim_lin = vae_arg_dict['latent_dim']
use_vm = False
use_cat = False
if 'latent_dim_vm' in vae_arg_dict:
  latent_dim_vm = vae_arg_dict['latent_dim_vm']
  use_vm = True
  print("Using VM mode")
else:
  print("Not using VM mode")
  latent_dim_vm = 0
if 'cat_dim' in vae_arg_dict:
  latent_dim_cat = vae_arg_dict['cat_dim']
  use_cat = True
  print("Using cat mode")
  if 'cat_priors' in vae_arg_dict:
    prob_a = np.array(vae_arg_dict['cat_priors'])
    a = prob_a / (1-prob_a)
  else:
    prob_a = np.logspace(-3,np.log10(0.5),32)
    a = prob_a / (1-prob_a)
else:
  print("Not using cat mode")
  latent_dim_cat = 0
latent_dim = latent_dim_lin + latent_dim_vm + latent_dim_cat

if args.img_prefix:
  file_prefix = args.img_prefix
else:
  file_prefix = model_dir + '/'
print("Saving files with prefix", file_prefix)




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


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# from colorspacious import cspace_converter
from collections import OrderedDict


from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class

import pandas

#import h5py
import pickle
#from scipy.stats import gaussian_kde

from pyjet import cluster
import re
import glob


if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")

if args.utils:
  import importlib
  VAE_model_tools = importlib.import_module(args.utils)
  build_and_compile_annealing_vae = getattr(VAE_model_tools,'build_and_compile_annealing_vae')
  betaVAEModel = getattr(VAE_model_tools,'betaVAEModel')
  reset_metrics = getattr(VAE_model_tools,'reset_metrics')
else:
  import utils.VAE_model_tools
  from utils.VAE_model_tools import build_and_compile_annealing_vae, betaVAEModel, reset_metrics


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


def Epxpypz_to_ptetaphiE(jets):

  E = jets[:,:,0]
  px = jets[:,:,1]
  py = jets[:,:,2]
  pz = jets[:,:,3]

  pt = np.sqrt(np.square(px) + np.square(py))
  phi = np.arctan2(py,px)
  eta = np.arcsinh(pz/pt)

  newjets = np.zeros(jets.shape)
  newjets[:,:,0] = pt
  newjets[:,:,1] = eta
  newjets[:,:,2] = phi
  newjets[:,:,3] = E

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

if args.boost:    
  def center_jets_ptetaphiE(jets):
    cartesian_jets = ptetaphiE_to_Epxpypz(jets)
    sumjet_cartesian = np.sum(cartesian_jets,axis=1)
    sumjet_phi = np.arctan2(sumjet_cartesian[:,2],sumjet_cartesian[:,1])

    phi_rotated_jets = np.copy(jets)
    phi_rotated_jets[:,:,2] = phi_rotated_jets[:,:,2] - sumjet_phi[:,None]
    phi_rotated_jets[:,:,2] = phi_rotated_jets[:,:,2] + np.pi
    phi_rotated_jets[:,:,2] = np.mod(phi_rotated_jets[:,:,2],2*np.pi)
    phi_rotated_jets[:,:,2] = phi_rotated_jets[:,:,2] - np.pi

    cartesian_jets = ptetaphiE_to_Epxpypz(phi_rotated_jets)
    sumjet_cartesian = np.sum(cartesian_jets,axis=1)

    E = sumjet_cartesian[:,0]
    pt = np.sqrt(np.square(sumjet_cartesian[:,1]) + np.square(sumjet_cartesian[:,2]))
    ptp = 550.
    beta = (E*pt - ptp*np.sqrt(np.square(E) - np.square(pt) + np.square(ptp)))/(np.square(E) + np.square(ptp))
    gamma = 1/np.sqrt(1-np.square(beta))

    ptnews = gamma[:,None]*(cartesian_jets[:,:,1] - beta[:,None]*cartesian_jets[:,:,0])
    Enews = gamma[:,None]*(cartesian_jets[:,:,0] - beta[:,None]*cartesian_jets[:,:,1])

    cartesian_jets[:,:,0] = Enews
    cartesian_jets[:,:,1] = ptnews
    sumjet_cartesian = np.sum(cartesian_jets,axis=1)


    sumjet_y = 0.5*np.log((sumjet_cartesian[:,0] + sumjet_cartesian[:,-1])/(sumjet_cartesian[:,0] - sumjet_cartesian[:,-1]))

    newjets = Epxpypz_to_ptetaphiE(cartesian_jets)

    ptyphim_jets = ptetaphiE_to_ptyphim(Epxpypz_to_ptetaphiE(cartesian_jets))
    ptyphim_jets[:,:,-1] = np.zeros(np.shape(ptyphim_jets[:,:,-1]))
    #print(ptyphim_jets[:3,:,:])

    transformed_jets = np.copy(ptyphim_jets)
    transformed_jets[:,:,1] = ptyphim_jets[:,:,1] - sumjet_y[:,None]

    transformed_jets[transformed_jets[:,:,0] == 0] = 0

    return ptyphim_to_ptetaphiE(transformed_jets)

else:
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


def kl_loss_bern_individual(r):
  delta = 1-r
  is_small = np.less(np.abs(delta), 1e-5)
  default_return = (1+r)*np.log(r)/(r-1) - 2
  return np.where(is_small,np.square(delta)/6,default_return)

def kl_loss_bern(log_alpha_bern,a):
  r = a/np.exp(log_alpha_bern)
  kl_separate = kl_loss_bern_individual(r)
  return kl_separate

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
        #plt.show()
        
def plot_KL_logvar(outs_array,xlim=None,ylim=None,showhist=False, numhists=10,hist_ylim=None,hist_xlim=None):

    if use_vm:
      y_pred ,z_mean, z_log_var, losses, _ = outs_array[0]
      KL = losses
    elif use_cat:
      y_pred ,z_mean, z_log_var, _, log_alpha_bern, _ = outs_array[0]

      KL = np.concatenate((kl_loss(z_mean, z_log_var),kl_loss_bern(log_alpha_bern,a)),axis=-1)
    else:
      y_pred ,z_mean, z_log_var, _ = outs_array[0]
      KL = kl_loss(z_mean, z_log_var)

    sort_kl = np.flip(np.argsort(np.mean(KL,axis=0)))

    rms_mean = np.sqrt(np.mean(np.square(z_mean),axis=0))

    if use_vm:
      plt.scatter(np.mean(KL[:,:latent_dim_lin],axis=0),rms_mean[:latent_dim_lin],s=5.)
      plt.scatter(np.mean(KL[:,latent_dim_lin:],axis=0),rms_mean[latent_dim_lin:],s=5.)
    else:
      plt.scatter(np.mean(KL[:,:latent_dim_lin],axis=0),rms_mean,s=5.)


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
        #plt.show()
        if hist_ylim:
            plt.ylim(hist_ylim)
        if hist_xlim:
            plt.xlim(hist_xlim)
    
    return sort_kl

def lab_to_p(lab):
  alpha = np.exp(lab)
  return alpha / (1+alpha)

# path to file
if args.parton:
  fn =  args.data_path + '/monoW-data-parton.h5'
  numparts = 2
  numtrain = 1500000
  print("Using parton data")
elif args.tj:
  fn =  args.data_path + '/train_processed.h5'
  numparts = 50
  numtrain = 1000000
  print("Using particle data")
elif args.top:
  fn =  args.data_path + '/t-data.h5'
  numparts = 50
  numtrain = 500000
  print("Using top data")
else:
  fn =  args.data_path + '/monoW-data-3.h5'
  numparts = 50
  numtrain = 500000
  print("Using particle data")

df = pandas.read_hdf(fn,stop=100000)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3)+sum(df.memory_usage(deep=True)) / (1024**3))

if args.tj:
  data = df.values[:,1:].reshape((-1,numparts,4))
  data[data == 1e5] = 0
  labels = df.values[:,0]
  tops = labels == 1
  js = labels == 0
else:
  data = df.values.reshape((-1,numparts,4))

if args.center:
  data = center_jets_ptetaphiE(data)

HT = np.sum(data[:,:,0],axis=-1)
data[:,:,0] = data[:,:,0]/HT[:,None]
data[:,:,-1] = data[:,:,-1]/HT[:,None]


if vae_arg_dict['num_inputs'] == 4:
  sig_input = np.zeros((len(data),numparts,4))
  sig_input[:,:,:2] = data[:,:,:2]
  sig_input[:,:,2] = np.cos(data[:,:,2])
  sig_input[:,:,3] = np.sin(data[:,:,2])
else:
  sig_input = np.zeros((len(data),numparts,5))
  sig_input[:,:,:2] = data[:,:,:2]
  sig_input[:,:,2] = np.cos(data[:,:,2])
  sig_input[:,:,3] = np.sin(data[:,:,2])
  sig_input[:,:,4] = np.log(data[:,:,3]+1e-8)

data_x = sig_input
data_y = data[:,:,:3]


train_x = data_x[:50000]
train_y = data_y[:50000]
valid_x = data_x[50000:]
valid_y = data_y[50000:]


train_output_dir = model_dir #create_dir(osp.join(output_dir, experiment_name))

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

print("Preparing to load weights")

def get_epoch(file):
    epoch = int(epoch_string.search(file).group()[1:-1])
    return epoch

def get_beta(file):
    beta = float(beta_string.search(file).group()[1:])
    return beta

epoch_string=re.compile('_\d*_')
beta_string=re.compile('_\d\.[\w\+-]*')
files = glob.glob(train_output_dir + '/model_weights_end*.hdf5')

if args.time_order:
  files.sort(key=os.path.getmtime)
  epochs = np.array([get_epoch(model_file) for model_file in files])
else:
  epochs = np.array([get_epoch(model_file) for model_file in files])
  sorted_args = np.argsort(epochs)
  files = [files[index] for index in sorted_args]
  epochs = epochs[sorted_args]


print("Found files:")
print(*files,sep='\n')



betas = np.array([get_beta(file) for file in files])

KLs = []
losses = []
recons = []


start=0

KLs_array = np.zeros((len(files[start:]), latent_dim))

for i, file in enumerate(files[start:]):
#     print("Loading", file)
    if i%10 == 0:
        print("Loading file", str(i), "of", str(len(files[start:])))
    vae.load_weights(file)
    vae.beta.assign(betas[i+start])
    outs_array = [vae.predict(train_x[:1000]) for j in range(1)]

    if use_vm:
      _, z_mean, z_log_var, kllosses, z = outs_array[0]
      KLs_array[i] = np.mean(kllosses,axis=0)
    elif use_cat:
      y_pred ,z_mean, z_log_var, _, log_alpha_bern, _ = outs_array[0]

      KLs_array[i] = np.mean(np.concatenate((kl_loss(z_mean, z_log_var),kl_loss_bern(log_alpha_bern,a)),axis=-1),axis=0)

      fig = plt.figure()
      plt.scatter(prob_a,np.mean(kl_loss_bern(log_alpha_bern,a),axis=0))
      plt.semilogx()
      plt.title('Epoch: ' + str(epochs[i+start]) + ', beta: ' + str(betas[i+start]))
      plt.xlabel('Prior')
      plt.ylabel('KL')
      plt.tight_layout()
      plt.savefig(file_prefix + 'KL_scatter_cat_' + str(i) + '_'+ str(betas[i+start]) + '.png')
      plt.close()

      fig = plt.figure()
      sort_kl_bern = np.argsort(np.mean(kl_loss_bern(log_alpha_bern,a),axis=0))
      plt.hist(log_alpha_bern[tops[:1000],sort_kl_bern[0]],bins = np.linspace(-15,15,100),histtype='step')
      plt.hist(log_alpha_bern[js[:1000],sort_kl_bern[0]],bins = np.linspace(-15,15,100),histtype='step')
      plt.title('Epoch: ' + str(epochs[i+start]) + ', beta: ' + str(betas[i+start]) + ', prior:' + a[sort_kl_bern[0]])
      plt.xlabel('Log Alpha')
      plt.tight_layout()
      plt.savefig(file_prefix + 'cathist_' + str(i) + '_'+ str(betas[i+start]) + '.png')
      plt.close()
      
    else:
      _, z_mean, z_log_var, z = outs_array[0]
      KLs_array[i] = np.mean(kl_loss(z_mean, z_log_var),axis=0)

    fig = plt.figure()
    plot_KL_logvar(outs_array,[-0.1,None],[-0.1,None])
    plt.title('Epoch: ' + str(epochs[i+start]) + ', beta: ' + str(betas[i+start]))
    plt.savefig(file_prefix + 'KL_scatter_' + str(i) + '_'+ str(betas[i+start]) + '.png')
    #plt.show()
    plt.close()

    result = vae.test_step([valid_x[:2000].astype(np.float32),valid_y[:2000].astype(np.float32)])
    
    losses += [result['loss'].numpy()]
    recons += [result['recon_loss'].numpy()]
    if use_vm:
      KLs += [result['KL loss'].numpy() + result['KL VM loss'].numpy()]
    elif use_cat:
      KLs += [result['KL loss'].numpy() + result['KL bern loss'].numpy()]
    else:
      KLs += [result['KL loss'].numpy()]

cmap = mpl.cm.get_cmap('viridis')

print(betas)
print(losses)
print(recons)
print(KLs)

np.savetxt(file_prefix + 'betas.dat',betas)
np.savetxt(file_prefix + 'losses.dat',losses)
np.savetxt(file_prefix + 'recons.dat',recons)
np.savetxt(file_prefix + 'KLs.dat',KLs)
np.savetxt(file_prefix + 'KLs_array.dat',KLs_array)

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
split_KLs_array = split_data(KLs_array)

print(split_betas)


def beta_to_betap(beta):
  return 500*beta
def betap_to_beta(betap):
  return betap/500

import math
n=int(math.ceil(len(split_betas)/2))
colors = [cmap(1.*i/(n) ) for i in range(n)]
# colors = ['C0','C1','C1','C2','C2']

fig=plt.figure()
plt.plot(betas)
plt.semilogy()
plt.title(args.img_title)
plt.xlabel('Iteration')
plt.ylabel(r'$\beta$')
plt.savefig(file_prefix +'betas.png')
#plt.show()
plt.close()

for i in range(len(split_betas)):
  fig = plt.figure()
  for j in range(latent_dim_lin):
    plt.plot(split_betas[i],split_KLs_array[i][:,j],color='C0')
  for j in range(latent_dim_lin,latent_dim_lin + latent_dim_vm):
    plt.plot(split_betas[i],split_KLs_array[i][:,j],color='C1')
  for j in range(latent_dim_lin + latent_dim_vm, latent_dim_lin + latent_dim_vm + latent_dim_cat):
    plt.plot(split_betas[i],split_KLs_array[i][:,j],color='C2')
  plt.semilogx()
  plt.xlim([1e-5,2.])
  plt.ylim([None,12.])
  ax = fig.axes[0]
  sec_ax = ax.secondary_xaxis('top',functions=(beta_to_betap,betap_to_beta))
  plt.title(args.img_title)
  plt.xlabel(r'$\beta$')
  plt.ylabel('KL')
  sec_ax.set_xlabel(r'$\hat{\beta}$  [GeV]')
  plt.tight_layout()
  plt.savefig(file_prefix +'all_KLs_' + str(i) + '.png')
  #plt.show()
  plt.close()

fig = plt.figure()
for i in range(len(split_betas)):
    style = '-'
    if i%2 == 1:
        style = '--'
    plt.plot(split_betas[i], split_losses[i],linestyle=style,color = colors[int(math.floor(i/2))])
plt.semilogy()
# plt.ylim([10,None])
plt.semilogx()
plt.xlabel(r'$\beta$')
plt.ylabel(r'Loss')
#plt.xlim(1e-2,1.)
ax = fig.axes[0]
sec_ax = ax.secondary_xaxis('top',functions=(beta_to_betap,betap_to_beta))
plt.title(args.img_title)
plt.savefig(file_prefix +'loss.png')
#plt.show()
plt.close()

fig = plt.figure()
for i in range(len(split_betas)):
    style = '-'
    if i%2 == 1:
        style = '--'
    plt.plot(split_betas[i], split_losses[i]*np.square(split_betas[i]),linestyle=style,color = colors[int(math.floor(i/2))])
plt.semilogy()
# plt.ylim([10,None])
plt.semilogx()
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\beta^2$ Loss')
ax = fig.axes[0]
sec_ax = ax.secondary_xaxis('top',functions=(beta_to_betap,betap_to_beta))
#plt.xlim(1e-2,1.)
#plt.ylim(1e-4,None)
plt.title(args.img_title)
plt.savefig(file_prefix +'losstimebetasqr.png')
#plt.show()
plt.close()

fig = plt.figure()
for i in range(len(split_betas)):
    style = '-'
    if i%2 == 1:
        style = '--'
    plt.plot(split_betas[i], split_recons[i],linestyle=style,color = colors[int(math.floor(i/2))])
plt.semilogy()
plt.semilogx()
plt.xlabel(r'$\beta$')
plt.ylabel(r'Recon Loss = EMD$^2$')
ax = fig.axes[0]
sec_ax = ax.secondary_xaxis('top',functions=(beta_to_betap,betap_to_beta))
#plt.ylim(1e-4,None)
#plt.xlim(1e-2,1.)
plt.title(args.img_title)
plt.tight_layout()
plt.savefig(file_prefix +'reconloss.png')
#plt.show()
plt.close()

fig = plt.figure()
for i in range(len(split_betas)):
    style = '-'
    if i%2 == 1:
        style = '--'
    plt.plot(split_betas[i], split_KLs[i],linestyle=style,color = colors[int(math.floor(i/2))])
#plt.semilogy()
plt.semilogx()
#plt.xlim(1e-2,1.)
#plt.ylim(0.1,None)
ax = fig.axes[0]
sec_ax = ax.secondary_xaxis('top',functions=(beta_to_betap,betap_to_beta))
plt.xlabel(r'$\beta$')
plt.ylabel(r'KL Loss')
plt.title(args.img_title)
plt.savefig(file_prefix +'KL.png')
#plt.show()
plt.close()



#fig = plt.figure()
#y_pred ,z_mean, z_log_var, losses, _ = outs_array[0]

#KL=kl_loss(z_mean, z_log_var)
#sort_kl = np.flip(np.argsort(np.mean(KL,axis=0)))

#rms_mean = np.sqrt(np.mean(np.square(z_mean),axis=0))

#plt.scatter(np.mean(KL,axis=0),rms_mean,s=5.)
#plt.xlim([-0.1,None])
#plt.ylim([-0.1,None])
#plt.xlabel('KL divergence')
#plt.ylabel(r'$\sqrt{\left\langle \mu^2 \right\rangle}$')
#plt.savefig(file_prefix + 'KL_scatter.png')
plt.close()



D1s = []
D2s = []
for j in range(len(split_betas)):
  style = '-'
  if i%2 == 1:
    style = '--'
  dKLs = np.array([split_KLs[j][i+1]-split_KLs[j][i] for i in range(len(split_KLs[j])-1)])
  dlogbetas = np.array([np.log(split_betas[j][i+1])-np.log(split_betas[j][i]) for i in range(len(split_KLs[j])-1)])
  D1 = -dKLs/dlogbetas
  D1s += [D1]
  drecons = np.array([split_recons[j][i+1]-split_recons[j][i] for i in range(len(split_recons[j])-1)])
  dbetasqrs = np.array([np.square(split_betas[j][i+1])-np.square(split_betas[j][i]) for i in range(len(split_KLs[j])-1)])
  D2 = drecons/dbetasqrs
  D2s += [D2]
  fig = plt.figure()
  plt.plot(split_betas[j][:-1],D1,linestyle=style)
  plt.plot(split_betas[j][:-1],D2,linestyle=style)
  plt.ylim([0,10])
  plt.xlim([1e-3,1.5])
  plt.semilogx()
  ax = fig.axes[0]
  plt.xlabel(r'$\beta$')
  plt.ylabel('Dimension')
  sec_ax = ax.secondary_xaxis('top',functions=(beta_to_betap,betap_to_beta))
  sec_ax.set_xlabel(r'$\hat{\beta}$  [GeV]')
  plt.title(args.img_title)
  plt.savefig(file_prefix +'Ds_' + str(j) + '.png')
  #plt.show()
  plt.close()
  
fig = plt.figure()
for j in range(len(split_betas)):
  style = '-'
  if i%2 == 1:
    style = '--'
  plt.plot(split_betas[j][:-1],D1s[j],color = colors[int(math.floor(j/2))],linestyle=style)
  plt.plot(split_betas[j][:-1],D2s[j],color = colors[int(math.floor(j/2))],linestyle=style)
  plt.ylim([0,10])
  plt.semilogx()
ax = fig.axes[0]
sec_ax = ax.secondary_xaxis('top',functions=(beta_to_betap,betap_to_beta))
plt.title(args.img_title)
plt.savefig(file_prefix +'Ds_all.png')
#plt.show()
plt.close()

pickle.dump( D1s, open(file_prefix + "D1s.p", "wb" ) )
pickle.dump( D2s, open(file_prefix + "D2s.p", "wb" ) )

print("Finished succesfully")
