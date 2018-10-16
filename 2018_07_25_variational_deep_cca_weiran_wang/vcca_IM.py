import numpy as np
import math
import tensorflow as tf
NUM_THREADS=2

class VCCA(object):
    
    def __init__(self, architecture, losstype1, losstype2, learning_rate=0.0001, l2_penalty=0.0, latent_penalty=1.0, STDVAR1=1.0, STDVAR2=1.0):

        # Save the architecture and parameters.
        self.network_architecture=architecture
        self.l2_penalty=l2_penalty
        self.learning_rate=tf.Variable(learning_rate,trainable=False)
        # self.learning_rate=learning_rate
        self.num_samples=L=5
        self.n_input1=n_input1=architecture["n_input1"]
        self.n_input2=n_input2=architecture["n_input2"]
        self.n_z=n_z=architecture["n_z"]
        self.n_h1=n_h1=architecture["n_h1"]
        self.n_h2=n_h2=architecture["n_h2"]
        # Trade-off parameter for KL divergence.
        self.latent_penalty=latent_penalty
        # Gaussian standard variation for the observation models of each view, only matters for losstype=2.
        self.STDVAR1=STDVAR1
        self.STDVAR2=STDVAR2
        
        # Tensorflow graph inputs.
        self.x1=tf.placeholder(tf.float32, [None, n_input1])
        self.x2=tf.placeholder(tf.float32, [None, n_input2])
        self.keepprob=tf.placeholder(tf.float32)

        # Variables to record training progress.
        self.epoch=tf.Variable(0, trainable=False)
        self.tunecost=tf.Variable(tf.zeros([1000]), trainable=False)
        
        # Initialize network weights and biases.
        initializer=tf.random_uniform_initializer(-0.05, 0.05)
        
        # Use the recognition network to obtain the Gaussian distribution (mean and log-variance) of latent codes.
        print("Building view 1 recognition network F ...")
        activation=self.x1
        width=n_input1
        with tf.variable_scope("F", reuse=None, initializer=initializer):
            for i in range(len(architecture["F_hidden_widths"])):
                print("\tLayer %d ..." % (i+1))
                activation=tf.nn.dropout(activation, self.keepprob)
                if i==(len(architecture["F_hidden_widths"])-1):
                    weights=tf.get_variable("weights_log_sigma_sq", [width, architecture["F_hidden_widths"][i]] )
                    biases=tf.get_variable("biases_log_sigma_sq", [architecture["F_hidden_widths"][i]] )
                    self.z_log_sigma_sq=tf.add(tf.matmul(activation, weights), biases)
                weights=tf.get_variable("weights_layer_" + str(i+1), [width, architecture["F_hidden_widths"][i]])
                biases=tf.get_variable("biases_layer_" + str(i+1), [architecture["F_hidden_widths"][i]])
                activation=tf.add(tf.matmul(activation, weights), biases)
                if not architecture["F_hidden_activations"][i] == None:
                    activation=architecture["F_hidden_activations"][i](activation)
                width=architecture["F_hidden_widths"][i]
        self.z_mean=activation
        
        
        # Private network for view 1.
        if n_h1>0:
            print("Building view 1 private network G1 ...")
            activation=self.x1
            width=n_input1
            with tf.variable_scope("G1", reuse=None, initializer=initializer):
                for i in range(len(architecture["G1_hidden_widths"])):
                    print("\tLayer %d ..." % (i+1))
                    activation=tf.nn.dropout(activation, self.keepprob)
                    if i==(len(architecture["G1_hidden_widths"])-1):
                        weights=tf.get_variable("weights_log_sigma_sq", [width, architecture["G1_hidden_widths"][i]] )
                        biases=tf.get_variable("biases_log_sigma_sq", [architecture["G1_hidden_widths"][i]] )
                        self.h1_log_sigma_sq=tf.add(tf.matmul(activation, weights), biases)
                    weights=tf.get_variable("weights_layer_" + str(i+1), [width, architecture["G1_hidden_widths"][i]])
                    biases=tf.get_variable("biases_layer_" + str(i+1), [architecture["G1_hidden_widths"][i]])
                    activation=tf.add(tf.matmul(activation, weights), biases)
                    if not architecture["G1_hidden_activations"][i] == None:
                        activation=architecture["G1_hidden_activations"][i](activation)
                    width=architecture["G1_hidden_widths"][i]
            self.h1_mean=activation
        
        
        # Private network for view 2.
        if n_h2>0:
            print("Building view 2 private network G2 ...")
            activation=self.x2
            width=n_input2
            with tf.variable_scope("G2", reuse=None, initializer=initializer):
                for i in range(len(architecture["G2_hidden_widths"])):
                    print("\tLayer %d ..." % (i+1))
                    activation=tf.nn.dropout(activation, self.keepprob)
                    if i==(len(architecture["G2_hidden_widths"])-1):
                        weights=tf.get_variable("weights_log_sigma_sq", [width, architecture["G2_hidden_widths"][i]] )
                        biases=tf.get_variable("biases_log_sigma_sq", [architecture["G2_hidden_widths"][i]] )
                        self.h2_log_sigma_sq=tf.add(tf.matmul(activation, weights), biases)
                    weights=tf.get_variable("weights_layer_" + str(i+1), [width, architecture["G2_hidden_widths"][i]])
                    biases=tf.get_variable("biases_layer_" + str(i+1), [architecture["G2_hidden_widths"][i]])
                    activation=tf.add(tf.matmul(activation, weights), biases)
                    if not architecture["G2_hidden_activations"][i] == None:
                        activation=architecture["G2_hidden_activations"][i](activation)
                    width=architecture["G2_hidden_widths"][i]
            self.h2_mean=activation
        
        
        # Calculate latent losses (KL divergence) for shared and private variables.
        latent_loss_z=- 0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)
        
        if n_h1>0:
            latent_loss_h1=- 0.5 * tf.reduce_sum(1 + self.h1_log_sigma_sq - tf.square(self.h1_mean) - tf.exp(self.h1_log_sigma_sq), 1)
        else:
            latent_loss_h1=tf.constant(0.0)
        
        if n_h2>0:
            latent_loss_h2=- 0.5 * tf.reduce_sum(1 + self.h2_log_sigma_sq - tf.square(self.h2_mean) - tf.exp(self.h2_log_sigma_sq), 1)
        else:
            latent_loss_h2=tf.constant(0.0)
        
        self.latent_loss=tf.reduce_mean(latent_loss_z + latent_loss_h1 + latent_loss_h2)
        
        
        # Draw L samples of z.
        z_epsshape=tf.multiply(tf.shape(self.z_mean), [L,1])
        eps=tf.random_normal(z_epsshape, 0, 1, dtype=tf.float32)
        self.z1=tf.add( tf.tile(self.z_mean, [L,1]), tf.multiply( tf.tile(tf.exp(0.5 * self.z_log_sigma_sq), [L,1]), eps))
        
        # Draw L samples of h1.
        if n_h1>0:
            h1_epsshape=tf.multiply(tf.shape(self.h1_mean), [L,1])
            eps=tf.random_normal(h1_epsshape, 0, 1, dtype=tf.float32)
            self.h1=tf.add( tf.tile(self.h1_mean, [L,1]), tf.multiply( tf.tile(tf.exp(0.5 * self.h1_log_sigma_sq), [L,1]), eps))
        
        # Use the generator network to reconstruct view 1.
        print("Building view 1 reconstruction network H1 ...")
        if n_h1>0:
            # activation=tf.concat(1, [self.z1, self.h1])
            activation=tf.concat([self.z1, self.h1],1)
            width=n_z + n_h1
        else:
            activation=self.z1
            width=n_z
            
        with tf.variable_scope("H1", reuse=None, initializer=initializer):
            for i in range(len(architecture["H1_hidden_widths"])):
                print("\tLayer %d ..." % (i+1))
                activation=tf.nn.dropout(activation, self.keepprob)
                if i==(len(architecture["H1_hidden_widths"])-1):
                    weights=tf.get_variable("weights_log_sigma_sq", [width, architecture["H1_hidden_widths"][i]])
                    biases=tf.get_variable("biases_log_sigma_sq", [architecture["H1_hidden_widths"][i]])
                    self.x1_reconstr_log_sigma_sq_from_z1=tf.add(tf.matmul(activation, weights), biases)
                weights=tf.get_variable("weights_layer_" + str(i+1), [width, architecture["H1_hidden_widths"][i]])
                biases=tf.get_variable("biases_layer_" + str(i+1), [architecture["H1_hidden_widths"][i]])
                activation=tf.add(tf.matmul(activation, weights), biases)
                if not architecture["H1_hidden_activations"][i] == None:
                    activation=architecture["H1_hidden_activations"][i](activation)
                width=architecture["H1_hidden_widths"][i]
        self.x1_reconstr_mean_from_z1=activation
        
        
        # Draw L samples of z.
        eps=tf.random_normal(z_epsshape, 0, 1, dtype=tf.float32)
        self.z2=tf.add( tf.tile(self.z_mean, [L,1]), tf.multiply( tf.tile(tf.exp(0.5 * self.z_log_sigma_sq), [L,1]), eps))

        # Draw L samples of h2.
        if n_h2>0:
            h2_epsshape=tf.multiply(tf.shape(self.h2_mean), [L,1])
            eps=tf.random_normal(h2_epsshape, 0, 1, dtype=tf.float32)
            self.h2=tf.add( tf.tile(self.h2_mean, [L,1]), tf.multiply( tf.tile(tf.exp(0.5 * self.h2_log_sigma_sq), [L,1]), eps))

        # Use the generator network to reconstruct view 2.
        print("Building view 2 reconstruction network H2 ...")
        if n_h2>0:
            # activation=tf.concat(1, [self.z2, self.h2])
            activation=tf.concat([self.z2, self.h2], 1)
            width=n_z + n_h2
        else:
            activation=self.z2
            width=n_z
        
        with tf.variable_scope("H2", reuse=None, initializer=initializer):
            for i in range(len(architecture["H2_hidden_widths"])):
                print("\tLayer %d ..." % (i+1))
                activation=tf.nn.dropout(activation, self.keepprob)
                if i==(len(architecture["H2_hidden_widths"])-1):
                    weights=tf.get_variable("weights_log_sigma_sq", [width, architecture["H2_hidden_widths"][i]])
                    biases=tf.get_variable("biases_log_sigma_sq", [architecture["H2_hidden_widths"][i]])
                    self.x2_reconstr_log_sigma_sq_from_z2=tf.add(tf.matmul(activation, weights), biases)
                weights=tf.get_variable("weights_layer_" + str(i+1), [width, architecture["H2_hidden_widths"][i]])
                biases=tf.get_variable("biases_layer_" + str(i+1), [architecture["H2_hidden_widths"][i]])
                activation=tf.add(tf.matmul(activation, weights), biases)
                if not architecture["H2_hidden_activations"][i] == None:
                    activation=architecture["H2_hidden_activations"][i](activation)
                width=architecture["H2_hidden_widths"][i]
        self.x2_reconstr_mean_from_z2=activation

        
        # Compute negative log-likelihood for input data.
        self.nll1=self._compute_reconstr_loss( tf.tile(self.x1, [L,1]), self.x1_reconstr_mean_from_z1, self.x1_reconstr_log_sigma_sq_from_z1, n_input1, losstype1, STDVAR1)

        self.nll2=self._compute_reconstr_loss( tf.tile(self.x2, [L,1]), self.x2_reconstr_mean_from_z2, self.x2_reconstr_log_sigma_sq_from_z2, n_input2, losstype2, STDVAR2)

        
        # Weight decay.
        self.weightdecay=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        
        # Define cost and use the ADAM optimizer.
        self.cost=latent_penalty * self.latent_loss + self.nll1 + self.nll2 + \
                   l2_penalty * self.weightdecay
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
        
        # Initializing the tensor flow variables and launch the session.
        init=tf.initialize_all_variables()
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))
        self.sess.run(init)


    def assign_lr(self, lr):
        self.sess.run(tf.assign(self.learning_rate, lr))

    def assign_epoch(self, EPOCH_VALUE):
        self.sess.run(tf.assign(self.epoch, EPOCH_VALUE))
    
    def assign_tunecost(self, TUNECOST_VALUE):
        self.sess.run(tf.assign(self.tunecost, TUNECOST_VALUE))


    def sample_bernoulli(self, MEAN):
        mshape=tf.shape(MEAN)
        return tf.select( tf.random_uniform(mshape) < MEAN, tf.ones(mshape), tf.zeros(mshape))

    
    def _compute_reconstr_loss(self, x_input, x_reconstr_mean, x_reconstr_log_sigma_sq, n_out, losstype, STDVAR):
        
        if losstype==0:
            # Cross entropy loss.
            reconstr_loss=- tf.reduce_sum(x_input * tf.log( 1e-6 + x_reconstr_mean ) + ( 1 - x_input ) * tf.log( 1e-6 + 1 - x_reconstr_mean), 1)
        elif losstype==1:
            # Least squares loss, with learned std.
            reconstr_loss=0.5 * tf.reduce_sum( tf.div( tf.square(x_input - x_reconstr_mean), 1e-6 + tf.exp(x_reconstr_log_sigma_sq) ), 1 ) + 0.5 * tf.reduce_sum( x_reconstr_log_sigma_sq, 1 ) + 0.5 * math.log(2 * math.pi) * n_out
        elif losstype==2:
            # Least squares loss, with specified std.
            reconstr_loss=0.5 * tf.reduce_sum( tf.square( (x_input - x_reconstr_mean)/STDVAR ), 1 ) + 0.5 * math.log(2 * math.pi * STDVAR * STDVAR) * n_out
        
        # Average over the minibatch.
        cost=tf.reduce_mean(reconstr_loss)
        return cost
    
    
    def partial_fit(self, X1, X2, keepprob):
        
        # Train model based on mini-batch of input data. Return cost of mini-batch.
        opt, cost=self.sess.run( [self.optimizer, self.cost], feed_dict={self.x1: X1, self.x2: X2, self.keepprob: keepprob})
        return cost

    
    def evaluate_cost(self, X1, X2):
        
        N=X1.shape[0]

        batchsize=5000
        cost=0.0
        cost_latent=0.0
        cost_nll1=0.0
        cost_nll2=0.0
        
        for batchidx in range(np.ceil(N / batchsize).astype(int)):
            
            idx=range( batchidx*batchsize, min(N, (batchidx+1)*batchsize) )
            tmpcost, tmpcost_latent, tmpcost_nll1, tmpcost_nll2=self.sess.run( [self.cost, self.latent_loss, self.nll1, self.nll2], feed_dict={self.x1: X1[idx,:], self.x2: X2[idx,:], self.keepprob: 1.0})

            cost+=tmpcost * len(idx)
            cost_latent+=tmpcost_latent * len(idx)
            cost_nll1+=tmpcost_nll1 * len(idx)
            cost_nll2+=tmpcost_nll2 * len(idx)
            
        return cost/N, cost_latent/N, cost_nll1/N, cost_nll2/N
    
    
    def transform_shared_minibatch(self, view, X):
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution.
        if view==1:
            return self.sess.run( [self.z_mean, tf.exp(0.5 * self.z_log_sigma_sq)], feed_dict={self.x1: X, self.keepprob: 1.0})
        else:
            raise ValueError("The shared variable is extracted from view 1!")    
    
    
    def transform_shared(self, view, X):
        
        N=X.shape[0]
        Din=X.shape[1]
        xtmp,_=self.transform_shared_minibatch(view, X[np.newaxis,0,:])
        Dout=xtmp.shape[1]
        
        Ymean=np.zeros([N, Dout], dtype=np.float32)
        Ystd=np.zeros([N, Dout], dtype=np.float32)
        batchsize=5000
        for batchidx in range(np.ceil(N / batchsize).astype(int)):
            idx=range( batchidx*batchsize, min(N, (batchidx+1)*batchsize) )
            tmpmean, tmpstd=self.transform_shared_minibatch(view, X[idx,:])
            Ymean[idx,:]=tmpmean
            Ystd[idx,:]=tmpstd
        return Ymean, Ystd
    
    
    def transform_private_minibatch(self, view, X):
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution.
        if view==1:
            return self.sess.run( [self.h1_mean, tf.exp(0.5 * self.h1_log_sigma_sq)], feed_dict={self.x1: X, self.keepprob: 1.0})
        else:
            return self.sess.run( [self.h2_mean, tf.exp(0.5 * self.h2_log_sigma_sq)], feed_dict={self.x2: X, self.keepprob: 1.0})
    
    
    def transform_private(self, view, X):
        
        N=X.shape[0]
        Din=X.shape[1]
        xtmp,_=self.transform_private_minibatch(view, X[np.newaxis,0,:])
        Dout=xtmp.shape[1]
        
        Ymean=np.zeros([N, Dout], dtype=np.float32)
        Ystd=np.zeros([N, Dout], dtype=np.float32)
        batchsize=5000
        for batchidx in range(np.ceil(N / batchsize).astype(int)):
            idx=range( batchidx*batchsize, min(N, (batchidx+1)*batchsize) )
            tmpmean, tmpstd=self.transform_private_minibatch(view, X[idx,:])
            Ymean[idx,:]=tmpmean
            Ystd[idx,:]=tmpstd
        return Ymean, Ystd
    
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution.
        if z_mu is None:
            z_mu=np.random.normal(size=self.n_z)

        # It does not matter whether we use self.z1 or self.z2 below as the generation networks are shared.
        x1_recon=dict()
        x1_recon["mean"]=self.sess.run( self.x1_reconstr_mean_from_z1, feed_dict={self.z1: z_mu, self.keepprob: 1.0})
        x1_recon["std"]=self.sess.run( tf.exp(0.5 * self.x1_reconstr_log_sigma_sq_from_z1), feed_dict={self.z1: z_mu, self.keepprob: 1.0})
        x2_recon=dict()
        x2_recon["mean"]=self.sess.run( self.x2_reconstr_mean_from_z2, feed_dict={self.z1: z_mu, self.keepprob: 1.0})
        x2_recon["std"]=self.sess.run( tf.exp(0.5 * self.x2_reconstr_log_sigma_sq_from_z2), feed_dict={self.z1: z_mu, self.keepprob: 1.0})

        return x1_recon, x2_recon
    
    
    def reconstruct(self, view, X1, X2):
        """ Use VCCA to reconstruct given data. """
        if view==1:
            x1_recon_mean, x1_recon_std=self.sess.run( [self.x1_reconstr_mean_from_z1, tf.exp(0.5 * self.x1_reconstr_log_sigma_sq_from_z1)], feed_dict={self.x1: X1, self.x2: X2, self.keepprob: 1.0})
            x2_recon_mean, x2_recon_std=self.sess.run( [self.x2_reconstr_mean_from_z2, tf.exp(0.5 * self.x2_reconstr_log_sigma_sq_from_z2)], feed_dict={self.x1: X1, self.x2: X2, self.keepprob: 1.0})
        else:
            x1_recon_mean, x1_recon_std=self.sess.run( [self.x1_reconstr_mean_from_z1, tf.exp(0.5 * self.x1_reconstr_log_sigma_sq_from_z1)], feed_dict={self.x1: X1, self.x2: X2, self.keepprob: 1.0})
            x2_recon_mean, x2_recon_std=self.sess.run( [self.x2_reconstr_mean_from_z2, tf.exp(0.5 * self.x2_reconstr_log_sigma_sq_from_z2)], feed_dict={self.x1: X1, self.x2: X2, self.keepprob: 1.0})

        x1_recon=dict();  x1_recon["mean"]=x1_recon_mean;  x1_recon["std"]=x1_recon_std
        x2_recon=dict();  x2_recon["mean"]=x2_recon_mean;  x2_recon["std"]=x2_recon_std
        
        return x1_recon, x2_recon
    
    
def train(model, trainData, tuneData, saver, checkpoint, batch_size=100, max_epochs=10, save_interval=5, keepprob=1.0):
    
    epoch=model.sess.run(model.epoch)
    TUNECOST=model.sess.run(model.tunecost)
    lr=model.sess.run(model.learning_rate)
    
    n_samples=trainData.num_examples
    total_batch=int(math.ceil(1.0 * n_samples / batch_size))
    
    # Training cycle.
    while epoch < max_epochs:
        print("Current learning rate %f" % lr)
        avg_cost=0.0
        
        # Loop over all batches.
        for i in range(total_batch):
            batch_x1, batch_x2, _=trainData.next_batch(batch_size)
            
            # Fit training using batch data.
            cost=model.partial_fit(batch_x1, batch_x2, keepprob)
            
            # Compute average loss.
            avg_cost +=cost / n_samples * batch_size

        # Compute validation error, turn off dropout.
        tune_cost, tune_latent, tune_nll1, tune_nll2=model.evaluate_cost( tuneData.images1, tuneData.images2 )
        TUNECOST[epoch]=tune_cost
        
        # Display logs per epoch step.
        epoch=epoch+1
        print("Epoch: %04d, train regret=%12.8f, tune cost=%12.8f" % (epoch, avg_cost, tune_cost))
        print("latent_loss=%.8f, nll1=%.8f, nll2=%.8f" % (tune_latent, tune_nll1, tune_nll2))
        # a=model.sess.run(model.z_mean, feed_dict={model.x1: tuneData._images1[1:10,:], model.x2: tuneData._images2[1:10,:], model.keepprob: 1.0})
        # print(a)
        
        if (checkpoint) and (epoch % save_interval == 0):
            if np.isnan(tune_cost):
                print("Loss is nan. Reverting to previously saved model ...")
                saver.restore(model.sess, checkpoint)
                epoch=model.sess.run(model.epoch)
                TUNECOST=model.sess.run(model.tunecost)
                lr=lr*0.5
                model.assign_lr(lr)
            else:
                model.assign_epoch(epoch)
                model.assign_tunecost(TUNECOST)
                save_path=saver.save(model.sess, checkpoint)
                print("Model saved in file: %s" % save_path)
            
    return model


"""


############################# Visualize shared information.
import matplotlib.pyplot as plt
%matplotlib qt

trainData,tuneData,testData=read_mnist()
x1_sample=testData.images1[::3]
x2_sample=testData.images2[::3]
y_sample=testData.labels[::3]
y_sample=np.reshape(y_sample, [y_sample.shape[0]])

z, _=model.transform_shared(1, x1_sample)

from sklearn.manifold import TSNE
tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=3000)
z_tsne=tsne.fit_transform( np.asfarray(z, dtype="float") )

COLORS=[
[1.0000, 0,      0     ],
[0,      1.0000, 0     ], 
[0,      0,      1.0000],
[1.0000, 0,      1.0000],
[0.9569, 0.6431, 0.3765],
[0.4000, 0.8039, 0.6667],
[0.5529, 0.7137, 0.8039],
[0.8039, 0.5882, 0.8039],
[0.7412, 0.7176, 0.4196],
[0,      0,      0     ]]

plt.figure(11,figsize=(10,10))
hhs=[]
for i in range(10):
    idx=np.argwhere(y_sample==(i+1))
    h=plt.plot(z_tsne[idx, 0], z_tsne[idx, 1], "o", c=COLORS[i], markersize=8.0)
    hhs.append(h[0])

plt.legend(hhs, ["1","2","3","4","5","6","7","8","9", "0"] )
plt.tight_layout()
plt.axis('off')
# plt.savefig("MNIST_VCCA_2D.eps")
plt.title("Shared")


############################# Visualized private information.
h1, _= model.transform_private(1, x1_sample)

from sklearn.manifold import TSNE
tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=3000)
h1_tsne=tsne.fit_transform( np.asfarray(h1, dtype="float") )

plt.figure(12,figsize=(10,10))
hhs=[]
for i in range(10):
    idx=np.argwhere(y_sample==(i+1))
    h=plt.plot(h1_tsne[idx, 0], h1_tsne[idx, 1], "o", c=COLORS[i], markersize=8.0)
    hhs.append(h[0])

plt.legend(hhs, ["1","2","3","4","5","6","7","8","9", "0"] )
plt.tight_layout()
plt.axis('off')
# plt.savefig("MNIST_VCCA_2D.eps")
plt.title("private")



############################# Visualized private information.
h2, _= model.transform_private(2, x2_sample)

from sklearn.manifold import TSNE
tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=3000)
h2_tsne=tsne.fit_transform( np.asfarray(h2, dtype="float") )

plt.figure(13,figsize=(10,10))
hhs=[]
for i in range(10):
    idx=np.argwhere(y_sample==(i+1))
    h=plt.plot(h2_tsne[idx, 0], h2_tsne[idx, 1], "o", c=COLORS[i], markersize=8.0)
    hhs.append(h[0])

plt.legend(hhs, ["1","2","3","4","5","6","7","8","9", "0"] )
plt.tight_layout()
plt.axis('off')
# plt.savefig("MNIST_VCCA_2D.eps")
plt.title("view 2 private")

import scipy.io as sio
sio.savemat('vcca_shared_emb_30.mat', {'z':z, 'z_tsne':z_tsne, 'h1':h1, 'h1_tsne':h1_tsne, 'h2':h2, 'h2_tsne':h2_tsne})



############################# VISUALIZE INPUTS.
tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=5000)
xinput=tsne.fit_transform_shared( np.asfarray(x1_sample, dtype="float") )

plt.figure(22,figsize=(10,10))
hhs=[]
for i in range(10):
    idx=np.argwhere(y_sample==(i+1))
    h=plt.plot(xinput[idx, 0], xinput[idx, 1], "o", c=COLORS[i])
    hhs.append(h[0])

plt.legend(hhs, ["1","2","3","4","5","6","7","8","0"] )
plt.tight_layout()
plt.savefig("INPUT1_tsne.eps")



############################# LINEAR SVM CLASSIFICATION.
trainData,tuneData,testData=read_mnist()
from sklearn import svm
lin_clf=svm.SVC(C=10, kernel="linear")

# train
svm_x_sample=trainData.images1[::]
svm_y_sample=np.reshape(trainData.labels[::], [svm_x_sample.shape[0]])
svm_z_sample, svm_z_std=model.transform_shared(1, svm_x_sample)
svm_z_sample_private, _=model.transform_private(1, svm_x_sample)
# svm_z_sample=np.concatenate([svm_z_sample, svm_z_sample_private], 1)
lin_clf.fit(svm_z_sample, svm_y_sample)

# predict
svm_x_sample=tuneData.images1
svm_y_sample=np.reshape(tuneData.labels, [svm_x_sample.shape[0]])
svm_z_sample, svm_z_std=model.transform_shared(1, svm_x_sample)
svm_z_sample_private, _=model.transform_private(1, svm_x_sample)
# svm_z_sample=np.concatenate([svm_z_sample, svm_z_sample_private], 1)
pred=lin_clf.predict(svm_z_sample)
np.mean(pred != svm_y_sample)

svm_x_sample=testData.images1
svm_y_sample=np.reshape(testData.labels, [svm_x_sample.shape[0]])
svm_z_sample, svm_z_std=model.transform_shared(1, svm_x_sample)
svm_z_sample_private, _=model.transform_private(1, svm_x_sample)
# svm_z_sample=np.concatenate([svm_z_sample, svm_z_sample_private], 1)
pred=lin_clf.predict(svm_z_sample)
np.mean(pred != svm_y_sample)

from sklearn.metrics import confusion_matrix
confusion_matrix(svm_y_sample, pred)
"""

"""
############################# Visualize train.
trainData,tuneData,testData=read_mnist()
x1_sample=trainData.images1[0::250]
x2_sample=trainData.images2[0::250]
x1_reconstruct, x2_reconstruct=model.reconstruct(1, x1_sample, x2_sample)

for i in range(0,200,20):
    # Plot one figure.
    plt.figure(1, figsize=(10.0,10.0))
    plt.imshow(np.transpose(x1_sample[i].reshape(28, 28)), vmin=0.0, vmax=1.0)
    # plt.title("view 1 input")
    plt.set_cmap("gray")
    plt.tight_layout()
    plt.axis("off")
    figname="shared_train_"+str(i)+"_view1_input.eps"
    plt.savefig(figname)
    raw_input()
    plt.close(1)

for i in range(0,200,20):
    plt.figure(2, figsize=(10.0,10.0))
    plt.imshow(np.transpose(x1_reconstruct["mean"][i].reshape(28, 28)), vmin=0, vmax=1)
    # plt.title("view 1 recons mean")
    plt.set_cmap("gray")
    plt.tight_layout()
    plt.axis("off")
    figname="shared_train_"+str(i)+"_view1_recon_mean.eps"
    plt.savefig(figname)
    raw_input()
    plt.close(2)

for i in range(0,200,20):
    plt.figure(3, figsize=(10.0,10.0))
    plt.imshow(np.transpose(x2_sample[i].reshape(28, 28)))
    # plt.title("view 2 input")
    plt.set_cmap("gray")
    plt.tight_layout()
    plt.axis("off")
    figname="shared_train_"+str(i)+"_view2_input.eps"
    plt.savefig(figname)
    raw_input()
    plt.close(3)

for i in range(0,200,20):
    plt.figure(4, figsize=(10.0,10.0))
    plt.imshow(np.transpose(x2_reconstruct["mean"][i].reshape(28, 28)))
    # plt.title("view 2 recons mean")
    plt.set_cmap("gray")
    plt.tight_layout()
    plt.axis("off")
    figname="shared_train_"+str(i)+"_view2_recon_mean.eps"
    plt.savefig(figname)
    raw_input()
    plt.close(4)

for i in range(0,200,20):
    plt.figure(5, figsize=(10.0,10.0))
    plt.imshow(np.transpose(x2_reconstruct["std"][i].reshape(28, 28)))
    # plt.title("view 2 recons std")
    plt.set_cmap("gray")
    plt.tight_layout()
    plt.axis("off")
    figname="shared_train_"+str(i)+"_view2_recon_std.eps"
    plt.savefig(figname)
    raw_input()
    plt.close(5)


"""


"""
# Script for XRMB.

import numpy as np
import tensorflow as tf
import vcca_shared as vcca

network_architecture=dict(
n_input1=273, # MNIST data input (img shape: 28*28)
n_input2=112, # MNIST data input (img shape: 28*28)
n_z=70,  # Dimensionality of latent space
F_hidden_widths=[1500, 1500, 1500, 70],
F_hidden_activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu, None],
G_hidden_widths=[1500, 1500, 1500, 70],
G_hidden_activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu, None],
H1_hidden_widths=[1500, 1500, 1500, 273],
H1_hidden_activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu, None],
H2_hidden_widths=[1500, 1500, 1500, 112],
H2_hidden_activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu, None]
)

from myreadinput import read_xrmb;
with tf.device("/cpu:0"):
    trainData,tuneData,testData=read_xrmb()
        
vcca=vcca.train(network_architecture, trainData, tuneData, losstype1=1, losstype2=1, learning_rate=0.001, batch_size=200, max_epochs=20, save_interval=1)

import scipy.io as sio
import numpy as np
data=sio.loadmat('/share/data/speech-multiview/wwang5/cca/XRMBf2KALDI_HTK_window7.mat');
proj, _=vcca.transform_shared(1, data['NEWMFCC'])
NEWMFCC=np.concatenate( (data['NEWMFCC'][:,117:156], proj), 1)
NEWLABEL=np.array( data['NEWLABEL'], dtype=np.float32)
sio.savemat('vccarecog.mat', {'NEWMFCC':NEWMFCC, 'NEWLABEL':NEWLABEL})

"""
