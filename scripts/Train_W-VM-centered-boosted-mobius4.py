import os
import os.path as osp
import sys
import json
import argparse
import glob

parser = argparse.ArgumentParser(description='Plot jets')
parser.add_argument('model_dir')
parser.add_argument('--model_file','--model_fn')

args = parser.parse_args()
print(args)
model_dir = args.model_dir
vae_args_file = model_dir + "/vae_args.dat"



if args.model_file == 'last':
  files = glob.glob(model_dir + '/model_weights_end*.hdf5')
  files.sort(key=os.path.getmtime)
  model_file = files[-1]
  with open(vae_args_file,'r') as f:
    vae_arg_dict = json.loads(f.read())

  print("\n\n vae_arg_dict:", vae_arg_dict)
elif args.model_file is not None:
  model_fn = args.model_file
  model_file = model_dir + '/' + model_fn
  print("Using model file", model_file)
  with open(vae_args_file,'r') as f:
    vae_arg_dict = json.loads(f.read())

  print("\n\n vae_arg_dict:", vae_arg_dict)
else:
  print("No model file specified, will train from beginning")
  model_file=None


import tensorflow as tf
# tf.config.experimental.set_visible_devices([], 'GPU')
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

import numpy as np

from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class
import utils.VAE_model_tools_mob4
from utils.VAE_model_tools_mob4 import build_and_compile_annealing_vae, betaVAEModel, reset_metrics, loss_tracker

import pandas

import h5py
import pickle

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
    
    # path to file
fn =  '/scratch/jcollins/monoW-data-parton.h5'
# fn =  '/media/jcollins/MAGIC!/monoW-data-3.h5'

# Option 1: Load everything into memory
df = pandas.read_hdf(fn,stop=2000000)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3)+sum(df.memory_usage(deep=True)) / (1024**3))


# Data file contains, for each event, 50 particles (with zero padding), each particle with pT, eta, phi, E.
data = df.values.reshape((-1,2,4))
data = center_jets_ptetaphiE(data)

# Normalize pTs so that HT = 1
HT = np.sum(data[:,:,0],axis=-1)
data[:,:,0] = data[:,:,0]/HT[:,None]
data[:,:,-1] = data[:,:,-1]/HT[:,None]

# Center jet (optional)

# Inputs x to NN will be: pT, eta, cos(phi), sin(phi), log E
# Separated phi into cos and sin for continuity around full detector, so make things easier for NN.
# Also adding the log E is mainly because it seems like it should make things easier for NN, since there is an exponential spread in particle energies.
# Feel free to change these choices as desired. E.g. px, py might be equally as good as pt, sin, cos.
sig_input = np.zeros((len(data),2,4))
sig_input[:,:,:2] = data[:,:,:2]
sig_input[:,:,2] = np.cos(data[:,:,2])
sig_input[:,:,3] = np.sin(data[:,:,2])
#sig_input[:,:,4] = np.log(data[:,:,3]+1e-8)


data_x = sig_input
# Event 'labels' y are [pT, eta, phi], which is used to calculate EMD to output which is also pT, eta, phi.
data_y = data[:,:,:3]


train_x = data_x[:1500000]
train_y = data_y[:1500000]
valid_x = data_x[1500000:1500000+100000]
valid_y = data_y[1500000:1500000+100000]

#output_dir = '/scratch/jcollins'

#experiment_name = 'W-parton-centered-mob'

train_output_dir = create_dir(model_dir)

if model_file is None:

  vae_arg_dict = {"encoder_conv_layers": [1024,1024,1024,1024],
                  "dense_size": [1024,1024,1024,1024],
                  "decoder_sizes": [1024,2048,1024,1024,1024],
                  "numIter": 10,
                  "reg_init": 1.,
                  "reg_final": 0.01,
                  "stopThr": 1e-3,
                  "num_inputs": 4,           # Size of x (e.g. pT, eta, sin, cos, log E)
                  "num_particles_in": 2,
                  "latent_dim": 1,
                  "latent_dim_vm": 1,
                  "verbose": 1,
                  "dropout": 0.2}
  
  print("Saving vae_arg_dict to",vae_args_file)
  print("\n",vae_arg_dict)
  
  with open(vae_args_file,'w') as file:
    file.write(json.dumps(vae_arg_dict))
    
  vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)
  
else:
  vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)
  print("Loading", model_file)

  vae.load_weights(model_file)

beta_set = np.logspace(-1.,-4,31)
betas = np.zeros(0)
for i in range(6):
  betas = np.append(betas, np.flip(beta_set[i:i+11:2]))
  betas = np.append(betas, beta_set[i+2:i+11:2])
for i in range(3,10):
  betas = np.append(betas, np.flip(beta_set[2*i:2*i + 11:2]))
  betas = np.append(betas, beta_set[2*i+2:2*i+11:2])

last_run_i = len(betas)
betas = np.append(betas, np.flip(beta_set)[1:])

print(betas)

init_epoch = 0
steps_per_epoch = 2000
batch_size = 32

reset_metrics_inst = reset_metrics()
reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5, verbose=1, mode='auto', min_delta=1e-4, \
cooldown=0, min_lr=1e-8)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0., patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)

start_i = 0
if args.model_file == 'last':
  import re
  start_i = len(files)
  def get_epoch(file):
    epoch = int(epoch_string.search(file).group()[1:-1])
    return epoch

  def get_beta(file):
    beta = float(beta_string.search(file).group())
    return beta

  epoch_string=re.compile('_\d*_')
  beta_string=re.compile('\d\.[\w\+-]*')

  init_epoch = get_epoch(model_file)

  print("Starting from epoch", init_epoch, ", and beta", betas[start_i])

for i in range(start_i,len(betas)):
    beta = betas[i]
    print("\n Changing beta to", beta)
    callbacks=[tf.keras.callbacks.CSVLogger(train_output_dir + '/log.csv', separator=",", append=True),
            reduceLR,earlystop,
#             modelcheckpoint,
            reset_metrics_inst]
    vae.beta.assign(beta)

    if i < last_run_i:
      K.set_value(vae.optimizer.lr,3e-5)
    else:
      K.set_value(vae.optimizer.lr,1e-5)

    my_history = vae.fit(x=train_x, y=train_y, batch_size=batch_size,
                epochs=10000,verbose=2,
                validation_data = (valid_x[:200*batch_size],valid_y[:200*batch_size]),
                callbacks = callbacks,
                initial_epoch=init_epoch,
                steps_per_epoch = steps_per_epoch
              )

    if np.isnan(loss_tracker.result().numpy()):
      if nan_counter > 10:
        print(nan_counter, "NaNs. Too many. Quitting.")
        quit()
      if last_save:
        print("Went Nan, reloading", last_save)
        nan_counter = nan_counter + 1
        vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)
        vae.fit(x=train_x[:1], y=train_y[:1], batch_size=1, epochs=1,verbose=2)
        vae.load_weights(last_save)
      else:
        print("Went nan but no last save, quitting...")
        quit()
    else:
      init_epoch = my_history.epoch[-1]
      i = i+1

    last_save = train_output_dir + '/model_weights_end_' + str(init_epoch) + '_' + "{:.1e}".format(beta) + '.hdf5'
    vae.save_weights(last_save)
