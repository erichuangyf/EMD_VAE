import tensorflow as tf
import tensorflow.keras as keras
# from tensorflow.python.keras.engine.training import enable_multi_worker
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten, Reshape, Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class
from tensorflow.python.keras.utils import tf_utils


import tensorflow_probability as tfp
tfkl = keras.layers
tfk = keras
tfd = tfp.distributions
tfpl = tfp.layers

from tensorflow_probability.python.distributions.von_mises import random_von_mises
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static

import numpy as np

class myVonMises(tfd.VonMises):
    @tf.function
    def _sample_n(self, n, seed=None):
        loc = tf.convert_to_tensor(self.loc)
        concentration = tf.convert_to_tensor(self.concentration)
        concentration = tf.broadcast_to(
            concentration, self._batch_shape_tensor(
                loc=loc, concentration=concentration))

        # random_von_mises does not work for zero concentration, so round it up to
        # something very small.
        tiny = np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny
        concentration = tf.maximum(concentration, tiny)

        sample_batch_shape = tf.concat([
            [n], prefer_static.shape(concentration)], axis=0)
        samples = random_von_mises(
            sample_batch_shape, concentration, dtype=self.dtype, seed=seed)

        gauss_sample = 1/(tf.math.sqrt(concentration)) * tf.random.normal(sample_batch_shape,dtype=self.dtype,seed=seed)
        where_large_conc = tf.greater(concentration, 1e6)
        

        samples = tf.where(where_large_conc,gauss_sample, samples)

        # vonMises(0, concentration) -> vonMises(loc, concentration)
        samples = samples + loc
        # Map the samples to [-pi, pi].
        samples = samples - 2. * np.pi * tf.round(samples / (2. * np.pi))
        return tf.cast(samples,tf.float32)  

loss_tracker = keras.metrics.Mean(name="loss")
recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
KL_loss_tracker = keras.metrics.Mean(name="KL_loss")
val_loss_tracker = keras.metrics.Mean(name="val_loss")
val_recon_loss_tracker = keras.metrics.Mean(name="val_recon_loss")
val_KL_loss_tracker = keras.metrics.Mean(name="val_KL_loss")

class betaVAEModel(keras.Model):

    def betaVAE_compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                recon_loss=None,
                KL_loss=None,
                KL_loss_bern=None,
                latent_dims_line = 1,
                latent_dims_circle = 1,
                use_dtype = tf.float32,
                **kwargs):

        self.compile(optimizer=optimizer,
                    #loss=loss,
                    metrics=metrics,
                    loss_weights=loss_weights,
                    sample_weight_mode=sample_weight_mode,
                    weighted_metrics=weighted_metrics,
                    **kwargs)
        self.recon_loss = recon_loss
        self.KL_loss = KL_loss
        self.KL_loss_bern = KL_loss_bern
        self.beta = tf.Variable(1.,trainable=False, name="beta")
        self.alpha = tf.Variable(1.,trainable=False, name="alpha")
        self.latent_dims_line = latent_dims_line
        self.latent_dims_circle = latent_dims_circle
        self.use_dtype=use_dtype
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.KL_loss_tracker = keras.metrics.Mean(name="KL_loss")
        self.KL_VM_loss_tracker = keras.metrics.Mean(name="KL_VM_loss")



    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred ,z_mean, z_log_var, losses, _ = self(x, training=True) # Forward pass
            # Compute our own loss
            recon_loss = self.recon_loss(y, y_pred)
            if self.latent_dims_line > 0:
                KL_loss = tf.reduce_mean(tf.reduce_sum(losses[:,:self.latent_dims_line],axis=-1))
            else:
                KL_loss = tf.constant(0.)
            if self.latent_dims_circle > 0:
                KL_loss_VM = tf.reduce_mean(tf.reduce_sum(losses[:,self.latent_dims_line:],axis=-1))
            else:
                KL_loss_VM = tf.constant(0.)
            loss = recon_loss/(2.*tf.square(tf.cast(self.beta,self.use_dtype))) + tf.cast(KL_loss,self.use_dtype) + tf.cast(self.alpha,self.use_dtype)*tf.cast(KL_loss_VM,self.use_dtype)
            # loss = recon_loss + tf.cast(KL_loss,self.use_dtype) *(tf.square(tf.cast(self.beta,self.use_dtype)))+ tf.cast(self.alpha,self.use_dtype)*tf.cast(KL_loss_VM,self.use_dtype)*(tf.square(tf.cast(self.beta,self.use_dtype)))


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.KL_loss_tracker.update_state(KL_loss)
        self.KL_VM_loss_tracker.update_state(KL_loss_VM)

        return {"loss": self.loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "KL loss": self.KL_loss_tracker.result(),
                "KL VM loss": self.KL_VM_loss_tracker.result(),
                "beta": self.beta,
                "alpha": self.alpha}

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred ,z_mean, z_log_var, losses, _ = self(x, training=False)  # Forward pass

        # Compute our own loss
        recon_loss = self.recon_loss(y, y_pred)
        if self.latent_dims_line > 0:
            KL_loss = tf.reduce_mean(tf.reduce_sum(losses[:,:self.latent_dims_line],axis=-1))
            # KL_loss = tf.reduce_mean(tf.reduce_sum(losses[:,:self.latent_dims_line],axis=-1))
        else:
            KL_loss = tf.constant(0.)
        if self.latent_dims_circle > 0:
            KL_loss_VM = tf.reduce_mean(tf.reduce_sum(losses[:,self.latent_dims_line:],axis=-1))
            # KL_loss_VM = tf.reduce_mean(tf.reduce_sum(losses[:,self.latent_dims_line:],axis=-1))
        else:
            KL_loss_VM = tf.constant(0.)
        

        loss = recon_loss/(2.*tf.square(tf.cast(self.beta,self.use_dtype))) + tf.cast(KL_loss,self.use_dtype) + tf.cast(self.alpha,self.use_dtype)*tf.cast(KL_loss_VM,self.use_dtype)
        # loss = recon_loss + tf.cast(KL_loss,self.use_dtype) *(tf.square(tf.cast(self.beta,self.use_dtype)))+ tf.cast(self.alpha,self.use_dtype)*tf.cast(KL_loss_VM,self.use_dtype)*(tf.square(tf.cast(self.beta,self.use_dtype)))

        self.loss_tracker.reset_states()
        self.recon_loss_tracker.reset_states()
        self.KL_loss_tracker.reset_states()
        self.KL_VM_loss_tracker.reset_states()

        self.loss_tracker.update_state(loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.KL_loss_tracker.update_state(KL_loss)
        self.KL_VM_loss_tracker.update_state(KL_loss_VM)


        return {"loss": self.loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "KL loss": self.KL_loss_tracker.result(),
                "KL VM loss": self.KL_VM_loss_tracker.result()}



# https://arxiv.org/pdf/1611.00712.pdf

def build_and_compile_annealing_vae(encoder_conv_layers = [256,256,256,256],
                                    dense_size = [256,256,256,256],
#                                    decoder = [512,256,256,256],
                                    decoder_sizes = [512,256,256,256],
                                    verbose=0,dropout=0,
                                    latent_dim = 128,
                                    latent_dim_vm = 128,
                                    optimizer=keras.optimizers.Adam(clipnorm=0.01,lr=0.0001),
                                    num_particles_out = 50,
                                    reg_init = 1.,
                                    reg_final = 0.01,
                                    numItermaxinner = 10,
                                    numIter = 10,
                                    stopThr=1e-3,
                                    temp = 0.3,
                                    EPSILON = 1e-6,
                                    num_particles_in = 100,
                                    check_err_period = 10,
                                    num_inputs = 4,
                                    use_dtype=tf.float32,
                                    renorm_clip = None,
                                    ):

   
    
    #Encoder
    inputs = tf.keras.Input(shape=(num_particles_in,num_inputs,), name='inputs')

    layer = inputs

    for layer_size in encoder_conv_layers:
        #layer = Conv1D(layer_size,1,bias_initializer='glorot_uniform')(layer)
        layer = Conv1D(layer_size,1)(layer)
        layer = keras.layers.LeakyReLU(0.1)(layer)
        if dropout > 0:
            layer = keras.layers.Dropout(dropout,noise_shape=(None,1,layer_size))(layer)
    
    # Sum layer
    layer = tf.keras.backend.sum(layer,axis=1)/np.sqrt(encoder_conv_layers[-1])

    # Dense layers
    for size in dense_size:
        #layer = Dense(size,bias_initializer='glorot_uniform')(layer)
        layer = Dense(size)(layer)
        layer = keras.layers.LeakyReLU(0.1)(layer)
        if dropout > 0:
            layer = keras.layers.Dropout(dropout)(layer)

#    layer = tf.keras.layers.BatchNormalization(
#        axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
#        beta_initializer='zeros', gamma_initializer='ones',
#        moving_mean_initializer='zeros',
#        moving_variance_initializer='ones', beta_regularizer=None,
#        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
#        renorm=True, renorm_clipping=renorm_clip, renorm_momentum=0.99, fused=None,
#        trainable=True, virtual_batch_size=None, adjustment=None, name=None
#    )(layer)

    z_mean = Dense(latent_dim, name='z_mean')(layer)
    z_log_var = Dense(latent_dim, name='z_log_var')(layer)

    gauss_layer = tf.stack([z_mean,z_log_var])
    z = tfpl.DistributionLambda(make_distribution_fn = lambda t: tfd.MultivariateNormalDiag(t[0],tf.exp(t[1]/2)),
                                    name="encoder_gauss_distribution"#,dtype=precision
                                )(gauss_layer)

    kl_loss_gauss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))



    vm_z_mean_x = tfkl.Dense(latent_dim_vm, name='encoder_vm_z_mean_x',activation=None)(layer)
    vm_z_mean_y = tfkl.Dense(latent_dim_vm, name='encoder_vm_z_mean_y',activation=None)(layer)

    vm_z_mean = tf.cast(tf.atan2(vm_z_mean_x,vm_z_mean_y, name = 'encoder_vm_z_mean'),tf.float64)
#     vm_z_log_var = tfkl.Dense(latent_dim_vm, name='encoder_vm_z_log_var',activation='elu',dtype=tf.float64)(layer)

#     concentration = 2*(tf.math.cosh(vm_z_log_var+1)-1+1e-6)

    vm_z_log_var = tfkl.Dense(latent_dim_vm, name='encoder_vm_z_log_var',dtype=tf.float64)(layer)

    concentration = tf.exp(vm_z_log_var)
    layer = tf.stack([vm_z_mean,concentration])

    vonmis = tfpl.DistributionLambda(make_distribution_fn = lambda t: myVonMises(t[0],t[1]),
                                    name="encoder_vm_distribution",dtype=tf.float64
                                    )(layer)

    if use_dtype is tf.float32:
        concentration64 = tf.cast(concentration,tf.float64)
    else:
        concentration64 = concentration
    i0e_concentration = tf.math.bessel_i0e(concentration64,name='i0e')
    i1e_concentration = tf.math.bessel_i1e(concentration64)
    conclimit = tf.cast(1/10.,tf.float64)
    concterm = tf.where(concentration64 > conclimit,tf.cast(0.,tf.float64),-tf.math.log(concentration64*conclimit)/2e+3)
    kl_loss_vm = tf.math.log(1 / i0e_concentration) + concentration64 * (i1e_concentration / i0e_concentration - 1) + concterm
        # #In large conc limit, I0e(conc) ~= 1/sqrt(2*pi*conc) , concentration64 * (i1e_concentration / i0e_concentration - 1) -> -0.5
    kl_loss_vm_approx = 0.5*tf.math.log(concentration64) + tf.cast(0.5*tf.math.log(2*np.pi),tf.float64) - 0.5
    use_approx = tf.greater(concentration64,1e6)
    kl_loss_vm = tf.where(use_approx,kl_loss_vm_approx,kl_loss_vm)
    if use_dtype is tf.float32:
        kl_loss_vm = tf.cast(kl_loss_vm,tf.float32)


        # centers = tfkl.Concatenate()([z_mean, tf.cast(vm_z_mean,tf.float32)])
        # log_vars = tfkl.Concatenate()([z_log_var, -tf.math.log(concentration)])
        # losses = [kl_loss_gauss, tf.cast(kl_loss_vm,tf.float32)]
        # samples = tfkl.Concatenate()([z,tf.cast(vonmis,tf.float32)])
    centers = tfkl.Concatenate()([z_mean, tf.cast(vm_z_mean,tf.float32)])
    log_vars = tfkl.Concatenate()([z_log_var, -tf.math.log(tf.cast(concentration,tf.float32))])
    losses = tfkl.Concatenate()([kl_loss_gauss, tf.cast(kl_loss_vm,tf.float32)])
    samples = tfkl.Concatenate()([z,vonmis])
    vonmis_cast = tf.cast(vonmis,tf.float32)
#     samples = vonmis

    encoder = Model(inputs, [centers,log_vars,losses,samples], name='encoder')

    if verbose > 1:
        encoder.summary()
    #plot_model(encoder, to_file='CNN-VAE_encoder.png', show_shapes=True)

    # Decoder
    latent_inputs = tfk.Input(shape=(latent_dim + latent_dim_vm,), name='z_sampling')
#     latent_inputs = tfk.Input(shape=(latent_dim_vm), name='z_sampling')

    layer = latent_inputs
    line_dims = latent_inputs[:,:latent_dim]
    circle_dims = latent_inputs[:,latent_dim:]

    circle_x = tf.sin(circle_dims)
    circle_y = tf.cos(circle_dims)

    layer = tfkl.Concatenate()([line_dims,circle_x,circle_y])

    
    for i, layer_size in enumerate(decoder_sizes):
        layer = Dense(layer_size)(layer)
        layer = keras.layers.LeakyReLU(0.1)(layer)
        if dropout > 0:
            layer = keras.layers.Dropout(dropout)(layer)


    layer = Dense(num_particles_out*4)(layer)
    layer = Reshape((num_particles_out,4))(layer)
    layer_pT = layer[:,:,0:1]
    layer_pT = tf.keras.layers.Softmax(axis=-2)(layer_pT)
    layer_eta = layer[:,:,1:2]
    layer_phi = tf.math.atan2(layer[:,:,3],layer[:,:,2])
    layer_phi = tf.expand_dims(layer_phi,axis=-1)
    decoded = tf.keras.layers.Concatenate()([layer_pT,layer_eta,layer_phi])

    decoder = Model(latent_inputs, decoded, name='decoder')
    if verbose > 1:
        decoder.summary()
    #plot_model(decoder, to_file='CNN-VAE_decoder.png', show_shapes=True)


    outputs = decoder(encoder(inputs)[3])
    vae = betaVAEModel(inputs, [outputs, centers,log_vars,losses,samples], name='VAE')


    sinkhorn_knopp_tf_inst = sinkhorn_knopp_tf_scaling_stabilized_class(reg_init,
                                                                            reg_final,
                                                                            numItermaxinner=numItermaxinner,
                                                                            numIter=numIter,
                                                                            stopThr=stopThr,
                                                                            check_err_period = check_err_period,
                                                                            dtype=tf.float64,
                                                                            sparse = False)   

    # @tf.function
    def return_return_loss(pt_outs, x_outs, pt_in, x_in):

        @tf.custom_gradient
        def return_loss(pt_out, x_out):

            epsilon = np.float64(1e-10)
            
            ground_distance = ground_distance_tf_nograd(x_in,x_out)

            match = sinkhorn_knopp_tf_inst(pt_in, pt_out, tf.stop_gradient(ground_distance))        
            recon_loss = tf.linalg.trace(tf.matmul(tf.stop_gradient(tf.cast(match,tf.float32)),ground_distance,transpose_b=True))
            
            def grad(dL):
                aones = tf.fill(tf.shape(pt_in),np.float64(1.))
                bones = tf.fill(tf.shape(pt_out),np.float64(1.))

                Mnew = tf.cast(tf.transpose(ground_distance,perm=[0,2,1]),tf.float64)

                T = tf.cast(tf.transpose(match,perm=[0,2,1]),tf.float64)
                Ttilde = T[:,:,:-1]

                L = T * Mnew
                Ltilde = L[:,:,:-1]

                D1 = tf.linalg.diag(tf.reduce_sum(T,axis=-1))
                D2 = tf.linalg.diag(1/(tf.reduce_sum(Ttilde,axis=-2) + np.float64(1e-100))) # Add epsilon to ensure invertibility

                H = D1 - tf.matmul(tf.matmul(Ttilde,D2),Ttilde,transpose_b=True) + epsilon* tf.eye(num_rows = tf.shape(bones)[-1],batch_shape = [tf.shape(bones)[0]],dtype=tf.float64) # Add small diagonal piece to make sure H is invertible in edge cases.

                f = - tf.reduce_sum(L,axis=-1) + tf.squeeze(tf.matmul(tf.matmul(Ttilde,D2),tf.expand_dims(tf.reduce_sum(Ltilde,axis=-2),-1)),axis=-1)
                g = tf.squeeze(tf.matmul(tf.linalg.inv(H),tf.expand_dims(f,-1)),axis=-1)

                grad_pT = g - bones*tf.expand_dims(tf.reduce_sum(g,axis=-1),-1)/tf.cast(tf.shape(bones)[1],tf.float64)
                
                grad_x_out = tf.gradients(recon_loss,x_out)[0]
                
                return [-tf.expand_dims(dL,-1) * tf.cast(grad_pT,tf.float32),
                        tf.expand_dims(tf.expand_dims(dL,-1),-1)*tf.cast(grad_x_out,tf.float32)]

            return recon_loss, grad
        return return_loss(pt_outs, x_outs)


    @tf.function
    def recon_loss(x, x_decoded_mean):
        pt_out = x_decoded_mean[:,:,0]
        x_out = x_decoded_mean[:,:,1:]
        pt_in = x[:,:,0]
        x_in = x[:,:,1:]
        return tf.reduce_mean(tf.square(return_return_loss(pt_out, x_out, pt_in, x_in)),axis=0)


  
    vae.betaVAE_compile(recon_loss=recon_loss,
                        optimizer=optimizer,
                        experimental_run_tf_function=False,
                        latent_dims_line=latent_dim,
                        latent_dims_circle=latent_dim_vm,
                        use_dtype=use_dtype#,
                        #metrics = [KL_loss_VM_func]
               )
    if verbose:
        vae.summary()
    
    return vae, encoder, decoder

class reset_metrics(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        loss_tracker.reset_states()
        recon_loss_tracker.reset_states()
        KL_loss_tracker.reset_states()
        val_loss_tracker.reset_states()
        val_recon_loss_tracker.reset_states()
        val_KL_loss_tracker.reset_states()


class myTerminateOnNaN(keras.callbacks.Callback):
  """Callback that terminates training when a NaN loss is encountered.
  """

  def __init__(self):
    super(myTerminateOnNaN, self).__init__()
    self._supports_tf_logs = True

  def on_epoch_end(self, batch, logs=None):
    logs = logs or {}
    loss = logs.get('loss')
    if loss is not None:
      loss = tf_utils.to_numpy_or_python_type(loss)
      if np.isnan(loss) or np.isinf(loss):
        print('Batch %d: Invalid loss, terminating training' % (batch))
        self.model.stop_training = True
