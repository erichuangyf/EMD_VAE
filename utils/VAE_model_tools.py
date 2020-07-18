import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.engine.training import enable_multi_worker



loss_tracker = keras.metrics.Mean(name="loss")
recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
KL_loss_tracker = keras.metrics.Mean(name="KL_loss")
KL_bern_loss_tracker = keras.metrics.Mean(name="KL_bern_loss")

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
    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred ,z_mean, z_log_var, z, alpha_bern, z_bern_sigmoid = self(x, training=True)  # Forward pass
            # Compute our own loss
            recon_loss = self.recon_loss(y, y_pred)
            KL_loss = self.KL_loss(z_mean, z_log_var)
            KL_loss_bern = self.KL_loss_bern(alpha_bern)

            loss = recon_loss/(tf.square(self.beta)) + KL_loss + self.alpha*KL_loss_bern


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        recon_loss_tracker.update_state(recon_loss)
        KL_loss_tracker.update_state(KL_loss)
        KL_bern_loss_tracker.update_state(KL_loss_bern)

        return {"loss": loss_tracker.result(),
                "recon_loss": recon_loss_tracker.result(),
                "KL loss": KL_loss_tracker.result(),
                "KL bern loss": KL_bern_loss_tracker.result()}



# https://arxiv.org/pdf/1611.00712.pdf

def build_and_compile_annealing_vae(encoder_conv_layers = [256,256,256,256],
                                    dense_size = 256,
                                    decoder = [512,256,256,256],
                                    verbose=0,dropout=0,
                                    latent_dim = 128,
                                    optimizer=keras.optimizers.Adam(),
                                    num_particles_out = 50,
                                    reg_init = 1.,
                                    reg_final = 0.01,
                                    numItermaxinner = 10,
                                    numIter = 10,
                                    stopThr=1e-3,
                                    temp = 0.3,
                                    EPSILON = 1e-6,
                                    num_particles_in = 100,
                                    check_err_period = 10):
    

    
    def sampling_gauss(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def sampling_bern(theta):

        batch = K.shape(theta)[0]
        dim = K.int_shape(theta)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_uniform(shape=(batch, dim),maxval=1-EPSILON,minval=EPSILON)
        #return (K.log(epsilon) - K.log(1-epsilon) + K.log(theta+EPSILON) - K.log(1-theta+EPSILON))/temp
        return (K.log(epsilon) - K.log(1-epsilon) + K.log(theta+EPSILON))/temp
    
    #Encoder
    inputs = tf.keras.Input(shape=(num_particles_in,4,), name='inputs')

    layer = inputs

    for layer_size in encoder_conv_layers:
        layer = Conv1D(layer_size,1,bias_initializer='glorot_uniform')(layer)
        layer = keras.layers.ReLU()(layer)
        if dropout > 0:
            layer = keras.layers.Dropout(dropout)(layer)
    
    # Sum layer
    layer = tf.keras.backend.sum(layer,axis=1)

    # Dense layers 1
    layer = Dense(dense_size,bias_initializer='glorot_uniform')(layer)
    layer = keras.layers.ReLU()(layer)
    layer = Dense(dense_size,bias_initializer='glorot_uniform')(layer)
    layer = keras.layers.ReLU()(layer)

    # Cat latent
    alpha_bern = Dense(cat_dim, name='alpha_bern',bias_initializer='glorot_uniform')(layer)
    alpha_bern = keras.layers.ELU()(alpha_bern)
    alpha_bern = keras.layers.Lambda(lambda x: x + 1)(alpha_bern)
#     alpha_bern = Activation('sigmoid')(alpha_bern)


    z_bern = Lambda(sampling_bern, output_shape=(cat_dim,), name='z_bern')(alpha_bern)
    z_bern_sigmoid = Activation('sigmoid')(z_bern)
    
    layer = tf.keras.layers.Concatenate()([layer,z_bern_sigmoid])

    # Dense layers 2
    layer = Dense(dense_size,bias_initializer='glorot_uniform')(layer)
    layer = keras.layers.ReLU()(layer)
    layer = Dense(dense_size,bias_initializer='glorot_uniform')(layer)
    layer = keras.layers.ReLU()(layer)
      
    z_mean = Dense(latent_dim, name='z_mean',bias_initializer='glorot_uniform')(layer)
    z_log_var = Dense(latent_dim, name='z_log_var',bias_initializer='glorot_uniform')(layer)
    z = Lambda(sampling_gauss, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z, alpha_bern, z_bern_sigmoid], name='encoder')
    if verbose:
        encoder.summary()
    #plot_model(encoder, to_file='CNN-VAE_encoder.png', show_shapes=True)

    # Decoder
    latent_inputs_gauss = Input(shape=(latent_dim,), name='z_sampling')
    latent_inputs_bern = Input(shape=(cat_dim,), name='z_sampling_bern')
    layer = tf.keras.layers.Concatenate()([latent_inputs_gauss,latent_inputs_bern])
    #layer = latent_inputs
    
    for i, layer_size in enumerate(decoder):
        layer = Dense(layer_size,bias_initializer='glorot_uniform')(layer)
        layer = keras.layers.ReLU()(layer)
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

    decoder = Model([latent_inputs_gauss,latent_inputs_bern], decoded, name='decoder')
    if verbose:
        decoder.summary()
    #plot_model(decoder, to_file='CNN-VAE_decoder.png', show_shapes=True)


    outputs = decoder([encoder(inputs)[2], encoder(inputs)[4]])
    vae = betaVAEModel(inputs, [outputs,z_mean, z_log_var, z, alpha_bern, z_bern_sigmoid], name='CNN-VAE')


    sinkhorn_knopp_tf_inst = sinkhorn_knopp_tf_scaling_stabilized_class(reg_init,
                                                                            reg_final,
                                                                            numItermaxinner=numItermaxinner,
                                                                            numIter=numIter,
                                                                            stopThr=stopThr,
                                                                            check_err_period = check_err_period,
                                                                            dtype=tf.float64,
                                                                            sparse = False)   

  
    def return_return_loss(pt_outs, x_outs, pt_in, x_in):
        @tf.custom_gradient
        def return_loss(pt_out, x_out):

            epsilon = np.float64(1e-10)
            ground_distance = ground_distance_tf_nograd(x_in,x_out)

            match = sinkhorn_knopp_tf_inst(pt_in, pt_out, tf.stop_gradient(ground_distance))        
            recon_loss = tf.linalg.trace(tf.matmul(tf.stop_gradient(match),ground_distance,transpose_b=True))
            
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


    @tf.function
    def kl_loss(z_mean, z_log_var):
        return tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1))


    prob_a = tf.constant(np.logspace(-3,np.log10(0.5),32))
    a = prob_a / (1-prob_a)

    @tf.custom_gradient
    def kl_loss_bern_individual(r):
        delta = 1-r
        is_small = tf.less(tf.abs(delta),1e-5)
        default_return = (1+r)*tf.math.log(r)/(r-1)
        to_return = tf.where(is_small,tf.square(delta)/6,default_return)

        def grad(dL):
            default_grad = (-1 + tf.square(r) - 2*r*tf.math.log(r))/(r*tf.square(r-1))
            return tf.where(is_small, delta/3, default_grad) * dL

        return to_return, grad

    @tf.function
    def kl_loss_bern(alpha_bern):
        r = a/alpha_bern
        kl_separate = kl_loss_bern_individual(r)
        kl_summed = tf.reduce_sum(kl_separate,axis=-1)
        return tf.reduce_mean(kl_summed)

    
    vae.betaVAE_compile(recon_loss=recon_loss,
                        KL_loss = kl_loss,
                        KL_loss_bern = kl_loss_bern,
                optimizer=optimizer,experimental_run_tf_function=False#,
                #metrics = [recon_loss,kl_loss(beta_input), kl_loss_bern(beta_input)]
               )
    
    vae.summary()
    
    return vae, encoder, decoder

