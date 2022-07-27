import os
import os.path as osp
import sys
import json
import argparse
import glob
import re
import gc

parser = argparse.ArgumentParser()
parser.add_argument('model_dir')
parser.add_argument('--model_file','--model_fn')
parser.add_argument('--parton',action='store_true')
parser.add_argument('--data_path',default='/scratch/jcollins')
parser.add_argument('--center',action='store_true')
parser.add_argument('--exponent',default=1.,type=float)

args = parser.parse_args()
print(args)
model_dir = args.model_dir
vae_args_file = model_dir + "/vae_args.dat"

init_epoch=0
start_i = 0
end_dropout = 120
if args.model_file == 'last':
  files = glob.glob(model_dir + '/model_weights_end*.hdf5')
  files.sort(key=os.path.getmtime)
  model_file = files[-1]
  with open(vae_args_file,'r') as f:
    vae_arg_dict = json.loads(f.read())

  print("\n\n vae_arg_dict:", vae_arg_dict)

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

  print("Starting from epoch", init_epoch)#, ", and beta", betas[start_i])

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
import utils.VAE_model_tools_param
from utils.VAE_model_tools_param import build_and_compile_annealing_vae, betaVAEModel, reset_metrics, loss_tracker, myTerminateOnNaN

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

model_dir = create_dir(model_dir)

from utils.jet_utils import center_jets_ptetaphiE

    # path to file
if args.parton:
  fn =  args.data_path + '/B_background.h5'
  numparts = 2
  print("Using parton data")
  numtrain = 1500000
else:
  fn =  args.data_path + '/B_background.h5'
  numparts = 50
  print("Using particle data")
  numtrain = 500000

print("Loading ", fn)
df = pandas.read_hdf(fn,stop=2000000)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3)+sum(df.memory_usage(deep=True)) / (1024**3))


# Data file contains, for each event, 50 particles (with zero padding), each particle with pT, eta, phi, E.
# we only need first 200th entries, weight is not required here
data = df.values[:,:200]
print(data.shape)
data = data.reshape((-1,numparts,4))
if args.center:
  data = center_jets_ptetaphiE(data)
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


train_x = data_x[:numtrain]
train_y = data_y[:numtrain]
valid_x = data_x[numtrain:numtrain+100000]
valid_y = data_y[numtrain:numtrain+100000]


#output_dir = '/scratch/jcollins'

#experiment_name = 'W-parton-centered-vm-lin2'
train_output_dir = create_dir(model_dir)
last_save = None

if model_file is None:


  if osp.exists(vae_args_file):
    print("Loading", vae_args_file)
    with open(vae_args_file,'r') as f:
      vae_arg_dict = json.loads(f.read())
  else:
    vae_arg_dict = {"encoder_conv_layers": [1024,1024,1024,1024],
                    "dense_size": [1024,1024,1024,1024],
                    "decoder_sizes": [1024,1024,1024,1024,1024],
                    "numIter": 10,
                    "reg_init": 1.,
                    "reg_final": 0.01,
                    "stopThr": 1e-3,
                    "num_inputs": 4,           # Size of x (e.g. pT, eta, sin, cos, log E)
                    "num_particles_in": numparts,
                    "latent_dim": 256,
                    "verbose": 1,
                    "dropout": 0.,
                    "exponent": args.exponent}

    print("Saving vae_arg_dict to",vae_args_file)
    print("\n",vae_arg_dict)

    with open(vae_args_file,'w') as file:
      file.write(json.dumps(vae_arg_dict))


  vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)

else:
#  if start_i < end_dropout:
#    vae_arg_dict["dropout"] = 0.1
  vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)
  vae.fit(x=train_x[:1], y=train_y[:1], batch_size=1, epochs=1,verbose=2)
  vae.load_weights(model_file)
  last_save = model_file

batch_size=100
reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=10, verbose=1, mode='auto', min_delta=1e-4, cooldown=0, min_lr=1e-8)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0., patience=20, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)
reset_metrics_inst = reset_metrics()

callbacks=[tf.keras.callbacks.CSVLogger(train_output_dir + '/log.csv', separator=",", append=True),
           reduceLR,
           earlystop,
           myTerminateOnNaN(),
           reset_metrics_inst]

logbeta_min = -6.
logbeta_max = 1.

beta_rs = np.concatenate((np.logspace(-5.,1.,13),
                          np.logspace(1.,-1.,5)[1:],
                          np.logspace(-1.,1.,5)[1:],
                          np.logspace(1.,-1.,5)[1:],
                          np.logspace(-1.,0.,3)[1:]))

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

steps_per_epoch = 1000
save_period = 10
nan_counter = 0

max_epoch_per_step = 100
#switch_max_epochs = len(beta_set_init)

for beta_r in beta_rs:
  print("\n\n Setting beta_r to", beta_r,"\n\n")

  K.set_value(vae.optimizer.lr,3e-5)
  vae.beta_r.assign(beta_r)

  my_history = vae.fit(x = generate_batch(train_x, train_y, batch_size,logbeta_min=logbeta_min,logbeta_max=logbeta_max),
                     epochs=init_epoch + max_epoch_per_step,verbose=1,
                       validation_data = generate_batch(valid_x,valid_y,batch_size*10,logbeta_min=logbeta_min,logbeta_max=logbeta_max),
                     validation_steps = 100,
                     callbacks = callbacks,
                     initial_epoch=init_epoch,
                     steps_per_epoch = steps_per_epoch
                   )

  init_epoch = my_history.epoch[-1]

  last_save = train_output_dir + '/model_weights_end_' + str(init_epoch) + '_betar_' + "{:.1e}".format(beta_r) + '.hdf5'
  vae.save_weights(last_save)

  gc.collect()
