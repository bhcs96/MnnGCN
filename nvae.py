# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 22:55:06 2019

@author: LIU-CY
"""


import tensorflow as tf
import numpy as np
from utils import weight_variable_glorot
from gae.layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
tfd = tf.contrib.distributions

class BSVGAE:
    def __init__(self, in_dim, n_batch=1, n_latent=5, var=False, x_dist="nb", n_node=1000, norm=0.5, pos_weight=10, batch_remove=False):
        self.in_dim = in_dim
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_node = n_node
        self.var = var
        self.x_dist = x_dist
        self.batch_remove = batch_remove
        self.norm = norm
        self.pos_weight = pos_weight
        
        if ((self.x_dist != "nb") & (self.x_dist != "mn")):
            print("Observation_distribution error.")
            return
        
        self.n_hidden_1 = 128
        self.weights={'encoder_h1': weight_variable_glorot(self.in_dim, self.n_hidden_1), 
                 'encoder_mu': weight_variable_glorot(self.n_hidden_1, self.n_latent),
                 'encoder_std': weight_variable_glorot(self.n_hidden_1, self.n_latent),
                 'decoder_h1':tf.Variable(tf.random_normal([self.n_latent, self.n_hidden_1])),
                 'decoder_h2':tf.Variable(tf.random_normal([self.n_hidden_1, self.in_dim]))}
        
        self.biases={'encoder_b1':tf.Variable(tf.random_normal([self.n_hidden_1])),
                'encoder_mu':tf.Variable(tf.random_normal([self.n_latent])),
                'encoder_std':tf.Variable(tf.random_normal([self.n_latent])),
                'decoder_b1':tf.Variable(tf.random_normal([self.n_latent])),
                'decoder_b2':tf.Variable(tf.random_normal([self.n_hidden_1]))}




        self.x = tf.placeholder(dtype=tf.float32, shape=[None, in_dim], name='x')
        #self.x = tf.sparse_placeholder(tf.float32)
        self.adj = tf.sparse_placeholder(tf.float32)
        self.adj_orig = tf.sparse_placeholder(tf.float32)
        self.batch_id = tf.placeholder(dtype=tf.int32, shape=[None], name='batch')
        #self.Count_in = tf.placeholder(dtype=tf.float32, shape=[None, in_dim], name='Count')
        self.library_size = tf.reduce_sum(self.x, axis=1, name='library-size')
        #, keepdims=True
        #self.Count_sum = tf.constant(count_sum)
        
        # Create prior distribution
        self.p_z = tfd.Normal(np.zeros(n_latent).astype(np.float32), np.ones(n_latent).astype(np.float32))
        
        # batch_id transform into one_hot
        self.batch = tf.one_hot(self.batch_id, self.n_batch)
        
        # vae encoder
        self.z_mu, self.z_log_std = self._encoder(self.x, self.adj)
        
        # posterior distribution
        self.q_z = tfd.Normal(self.z_mu, tf.exp(self.z_log_std))
        
        # sample from posterior
        self.z = self.q_z.sample()
        #self.z = self.z_mu + tf.random_normal([self.n_node, self.n_latent]) * tf.exp(self.z_log_std)
        # decoder
        self.mu, self.sigma_square, self.re_adj = self._decoder(self.z)
        #self.re_adj = self._decoder(self.z)
        # loss
        if self.x_dist == "mn":
            self.re_loss = -tfd.Multinomial(total_count=self.library_size, probs=self.mu).log_prob(self.x)
        elif self.x_dist == "nb":
            eps = 1e-16
            log_mu_sigma = tf.math.log(self.mu + self.sigma_square + eps)
            self.nb_loss = tf.math.lgamma(self.x + self.sigma_square) - tf.math.lgamma(self.sigma_square) - \
            tf.math.lgamma(self.x + 1) + self.sigma_square * tf.math.log(self.sigma_square + eps) - \
            self.sigma_square * log_mu_sigma + self.x * tf.math.log(self.mu + eps) - self.x * log_mu_sigma
            re_loss = tf.reduce_sum(self.nb_loss, axis=-1)
            self.re_loss = - tf.reduce_mean(re_loss)
        else:
            self.re_loss = 0

        self.re_loss_adj = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.reshape(tf.sparse_tensor_to_dense(self.adj_orig, validate_indices=False), [-1]), logits=tf.reshape(self.re_adj, [-1]), pos_weight=self.pos_weight))
        self.kl_loss = tf.reduce_mean(tf.reduce_sum(tfd.kl_divergence(self.q_z, self.p_z), axis=1))
        #self.kl_loss = -(0.5 / self.n_node) * tf.reduce_mean(tf.reduce_sum(1 + 2 * self.z_log_std - tf.square(self.z_mu) - tf.square(tf.exp(self.z_log_std)), 1))
        self.elbo = self.kl_loss + self.re_loss #+ self.re_loss_adj
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.elbo)
    
    
    def _encoder(self, x, adj):
        x = tf.math.log1p(x)
        x = tf.nn.l2_normalize(x, axis=-1)
        #x = sparse_to_tuple(x)
        #e1 = tf.matmul(x, self.weights["encoder_h1"])
        #x = tf.nn.dropout(x, tf.constant(1.))
        #e1 = tf.sparse_tensor_dense_matmul(x, self.weights["encoder_h1"])
        e1 = tf.add(tf.matmul(x, self.weights["encoder_h1"]), self.biases["encoder_b1"])
        e1 = tf.sparse_tensor_dense_matmul(self.adj, e1)
        e1 = tf.nn.relu(e1)
        
        #z_mu = tf.matmul(e1, self.weights["encoder_mu"])
        z_mu = tf.add(tf.matmul(e1, self.weights["encoder_mu"]), self.biases["encoder_mu"])
        z_mu = tf.sparse_tensor_dense_matmul(self.adj, z_mu)
        
        if self.var:
            z_log_std = tf.matmul(e1, self.weights["encoder_std"])
            z_log_std = tf.sparse_tensor_dense_matmul(self.adj, z_log_std)
            z_log_std = tf.nn.softplus(z_log_std)
        else:
            z_log_std =  tf.constant(np.zeros(self.n_latent).astype(np.float32))
        
        #a_pos = tf.multiply(tf.nn.softmax(e3),M)
        #return a_pos
        return z_mu, z_log_std
    
   
    
    def _decoder(self, z):
        #d0 = tf.layers.dense(inputs=z, units=32, activation=tf.nn.relu)
        d1 = tf.layers.dense(inputs=z, units=128, activation=tf.nn.relu)
        #d2 = tf.layers.dense(inputs=d1, units=512, activation=tf.nn.relu)
        mu = tf.layers.dense(inputs=d1, units=self.in_dim, activation=tf.nn.softmax)
        
        if self.x_dist == "nb":
            mu *= tf.expand_dims(self.library_size, dim=1)
            sigma_square = tf.layers.dense(inputs=d1, units=self.in_dim, activation=tf.nn.softplus)
            sigma_square = tf.reduce_mean(sigma_square, axis=0)
        else:
            sigma_square = None
        #p_re = tf.nn.softmax(logits)
        re_adj = tf.reshape(tf.matmul(z, tf.transpose(z)),[-1])   
        return mu, sigma_square, re_adj
    
    
    
    def _nb_loss(self, x, mu, sigma, eps=1e-16):
        log_mu_sigma = tf.math.log(mu + sigma + eps)
        ll = tf.math.lgamma(x + sigma) - tf.math.lgamma(sigma) - \
        tf.math.lgamma(x + 1) + sigma * tf.math.log(sigma + eps) - \
        sigma * log_mu_sigma + x * tf.math.log(mu + eps) - x * log_mu_sigma
        nb_loss = tf.reduce_sum(ll, axis=1)
        return nb_loss
        
        
    def vae_loss(self, X_in, z_pos, logits):
        ### lOSS Function
        # Multi Nominal
        #x_re_mn = tfd.Multinomial(p_re)
        #re_loss = x_mn.
        re_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, X_in))
        kl_loss = tfd.kl_divergence(z_pos, self.z_pri)
        loss = re_loss + kl_loss
        return loss
    
    

    
class BVAE:
    def __init__(self, in_dim, n_batch=1, n_latent=5, var=False, x_dist="nb", batch_remove=False):
        self.in_dim = in_dim
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.var = var
        self.x_dist = x_dist
        self.batch_remove = batch_remove
        
        if ((self.x_dist != "nb") & (self.x_dist != "mn")):
            print("Observation_distribution error.")
            return
        
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, in_dim], name='x')
        self.batch_id = tf.placeholder(dtype=tf.int32, shape=[None], name='batch')
        #self.Count_in = tf.placeholder(dtype=tf.float32, shape=[None, in_dim], name='Count')
        self.library_size = tf.reduce_sum(self.x, axis=1, name='library-size')
        #, keepdims=True
        #self.Count_sum = tf.constant(count_sum)
        
        # Create prior distribution
        self.p_z = tfd.Normal(np.zeros(n_latent).astype(np.float32), np.ones(n_latent).astype(np.float32))
        
        # batch_id transform into one_hot
        self.batch = tf.one_hot(self.batch_id, self.n_batch)
        
        # vae encoder
        self.z_mu, self.z_log_std = self._encoder(self.x, self.batch)
        
        # posterior distribution
        self.q_z = tfd.Normal(self.z_mu, tf.exp(self.z_log_std))
        
        # sample from posterior
        self.z = self.q_z.sample()
        
        # decoder
        self.mu, self.sigma_square = self._decoder(self.z, self.batch)
    
        # loss
        #self.re_loss = -tfd.NegativeBinomial(total_count=self.library_size, probs=self.mu, ).log_prob(self.x)
        #self.elbo = self._calculate_elbo()
        if self.x_dist == "mn":
            self.re_loss = -tfd.Multinomial(total_count=self.library_size, probs=self.mu).log_prob(self.x)
        elif self.x_dist == "nb":
            eps = 1e-16
            log_mu_sigma = tf.math.log(self.mu + self.sigma_square + eps)
            self.nb_loss = tf.math.lgamma(self.x + self.sigma_square) - tf.math.lgamma(self.sigma_square) - \
            tf.math.lgamma(self.x + 1) + self.sigma_square * tf.math.log(self.sigma_square + eps) - \
            self.sigma_square * log_mu_sigma + self.x * tf.math.log(self.mu + eps) - self.x * log_mu_sigma
            self.re_loss = -tf.reduce_sum(self.nb_loss, axis=-1)
            #self.re_loss = self._nb_loss()
        else:
            self.re_loss = 0
        
        
        self.kl_loss = tf.reduce_sum(tfd.kl_divergence(self.q_z, self.p_z), axis=1)
        self.elbo = tf.reduce_mean((self.re_loss + self.kl_loss))
        
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.elbo)
    
    
    def _encoder(self, x, adj):
        x = tf.math.log1p(x)
        x = tf.nn.l2_normalize(x, axis=-1)
        #x = sparse_to_tuple(x)
        #e1 = tf.matmul(x, self.weights["encoder_h1"])
        #x = tf.nn.dropout(x, tf.constant(1.))
        #e1 = tf.sparse_tensor_dense_matmul(x, self.weights["encoder_h1"])
        e1 = tf.add(tf.matmul(x, self.weights["encoder_h1"]), self.biases["encoder_b1"])
        e1 = tf.sparse_tensor_dense_matmul(self.adj, e1)
        e1 = tf.nn.relu(e1)
        
        #z_mu = tf.matmul(e1, self.weights["encoder_mu"])
        z_mu = tf.add(tf.matmul(e1, self.weights["encoder_mu"]), self.biases["encoder_mu"])
        z_mu = tf.sparse_tensor_dense_matmul(self.adj, z_mu)
        
        if self.var:
            z_log_std = tf.matmul(e1, self.weights["encoder_std"])
            z_log_std = tf.sparse_tensor_dense_matmul(self.adj, z_log_std)
            #z_log_std = tf.nn.softplus(z_log_std)
        else:
            z_log_std =  tf.constant(np.zeros(self.n_latent).astype(np.float32))
        
        #a_pos = tf.multiply(tf.nn.softmax(e3),M)
        #return a_pos
        return z_mu, z_log_std
    
    
    def _decoder(self, z, batch):
        if self.batch_remove:
            z = tf.concat([z, batch], 1)
        d0 = tf.layers.dense(inputs=z, units=32, activation=tf.nn.relu)
        d1 = tf.layers.dense(inputs=d0, units=128, activation=tf.nn.relu)
        #d2 = tf.layers.dense(inputs=d1, units=512, activation=tf.nn.relu)
        mu = tf.layers.dense(inputs=d1, units=self.in_dim, activation=tf.nn.softmax)
        
        if self.x_dist == "nb":
            mu *= tf.expand_dims(self.library_size, dim=1)
            sigma_square = tf.layers.dense(inputs=d1, units=self.in_dim, activation=tf.nn.softplus)
            sigma_square = tf.reduce_mean(sigma_square, axis=0)
        else:
            sigma_square = None
        #p_re = tf.nn.softmax(logits)
        return mu, sigma_square
    
    
    def _nb_loss(self, x, mu, sigma, eps=1e-16):
        log_mu_sigma = tf.math.log(mu + sigma + eps)
        ll = tf.math.lgamma(x + sigma) - tf.math.lgamma(sigma) - \
        tf.math.lgamma(x + 1) + sigma * tf.math.log(sigma + eps) - \
        sigma * log_mu_sigma + x * tf.math.log(mu + eps) - x * log_mu_sigma
        nb_loss = tf.reduce_sum(ll, axis=1)
        return nb_loss
             
    def vae_loss(self, X_in, z_pos, logits):
        ### lOSS Function
        # Multi Nominal
        #x_re_mn = tfd.Multinomial(p_re)
        #re_loss = x_mn.
        re_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, X_in))
        kl_loss = tfd.kl_divergence(z_pos, self.z_pri)
        loss = re_loss + kl_loss
        return loss
    
    
class BVGAE:
    def __init__(self, in_dim, n_batch=1, n_latent=5, var=False, x_dist="nb", n_node=1000, norm=0.5, pos_weight=10, batch_remove=False):
        self.in_dim = in_dim
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_node = n_node
        self.var = var
        self.x_dist = x_dist
        self.batch_remove = batch_remove
        self.norm = norm
        self.pos_weight = pos_weight
        
        if ((self.x_dist != "nb") & (self.x_dist != "mn")):
            print("Observation_distribution error.")
            return
        
        self.n_hidden_1 = 256
        self.weights={'encoder_h1': weight_variable_glorot(self.in_dim, self.n_hidden_1), 
                 'encoder_mu': weight_variable_glorot(self.n_hidden_1, self.n_latent),
                 'encoder_std': weight_variable_glorot(self.n_hidden_1, self.n_latent),
                 'decoder_h1':tf.Variable(tf.random_normal([self.n_latent, self.n_hidden_1])),
                 'decoder_h2':tf.Variable(tf.random_normal([self.n_hidden_1, self.in_dim]))}
        
        self.biases={'encoder_b1':tf.Variable(tf.random_normal([self.n_hidden_1])),
                'encoder_mu':tf.Variable(tf.random_normal([self.n_latent])),
                'encoder_std':tf.Variable(tf.random_normal([self.n_latent])),
                'decoder_b1':tf.Variable(tf.random_normal([self.n_latent])),
                'decoder_b2':tf.Variable(tf.random_normal([self.n_hidden_1]))}




        self.x = tf.placeholder(dtype=tf.float32, shape=[None, in_dim], name='x')
        #self.x = tf.sparse_placeholder(tf.float32)
        self.adj = tf.sparse_placeholder(tf.float32)
        self.adj_orig = tf.sparse_placeholder(tf.float32)
        self.batch_id = tf.placeholder(dtype=tf.int32, shape=[None], name='batch')
        #self.Count_in = tf.placeholder(dtype=tf.float32, shape=[None, in_dim], name='Count')
        #self.library_size = tf.reduce_sum(self.x, axis=1, name='library-size')
        #, keepdims=True
        #self.Count_sum = tf.constant(count_sum)
        
        # Create prior distribution
        self.p_z = tfd.Normal(np.zeros(n_latent).astype(np.float32), np.ones(n_latent).astype(np.float32))
        
        # batch_id transform into one_hot
        self.batch = tf.one_hot(self.batch_id, self.n_batch)
        
        # vae encoder
        self.z_mu, self.z_log_std = self._encoder(self.x, self.adj)
        
        # posterior distribution
        self.q_z = tfd.Normal(self.z_mu, tf.exp(self.z_log_std))
        
        # sample from posterior
        self.z = self.q_z.sample()
        #self.z = self.z_mu + tf.random_normal([self.n_node, self.n_latent]) * tf.exp(self.z_log_std)
        # decoder
        #self.mu, self.sigma_square = self._decoder(self.z, self.batch)
        self.re_adj = self._decoder(self.z)
        # loss
        #self.re_loss = -tfd.NegativeBinomial(total_count=self.library_size, probs=self.mu, ).log_prob(self.x)
        #self.elbo = self._calculate_elbo()
        self.re_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.reshape(tf.sparse_tensor_to_dense(self.adj_orig, validate_indices=False), [-1]), logits=self.re_adj, pos_weight=self.pos_weight))
        
        #self.kl_loss = -(0.5 / self.n_node) * tf.reduce_mean(tf.reduce_sum(1 + 2 * self.z_log_std - tf.square(self.z_mu) - tf.square(tf.exp(self.z_log_std)), 1))
        self.kl_loss = 1. / self.n_node * tf.reduce_mean(tf.reduce_sum(tfd.kl_divergence(self.q_z, self.p_z), axis=1))
        #self.re_loss = tf.reshape(self.re_adj, [-1])
        #self.elbo = tf.reduce_mean((self.re_loss + self.kl_loss))
        #self.elbo = tf.reduce_mean((self.kl_loss))/self.n_node  + self.norm * self.re_loss
        self.elbo = self.kl_loss  + self.norm * self.re_loss
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.elbo)
    
    
    def _encoder(self, x, adj):
        x = tf.math.log1p(x)
        x = tf.nn.l2_normalize(x, axis=-1)
        #x = sparse_to_tuple(x)
        #e1 = tf.matmul(x, self.weights["encoder_h1"])
        #x = tf.nn.dropout(x, tf.constant(1.))
        #e1 = tf.sparse_tensor_dense_matmul(x, self.weights["encoder_h1"])
        e1 = tf.add(tf.matmul(x, self.weights["encoder_h1"]), self.biases["encoder_b1"])
        e1 = tf.sparse_tensor_dense_matmul(self.adj, e1)
        e1 = tf.nn.relu(e1)
        
        #z_mu = tf.matmul(e1, self.weights["encoder_mu"])
        z_mu = tf.add(tf.matmul(e1, self.weights["encoder_mu"]), self.biases["encoder_mu"])
        z_mu = tf.sparse_tensor_dense_matmul(self.adj, z_mu)
        
        if self.var:
            z_log_std = tf.matmul(e1, self.weights["encoder_std"])
            z_log_std = tf.sparse_tensor_dense_matmul(self.adj, z_log_std)
            #z_log_std = tf.nn.softplus(z_log_std)
        else:
            z_log_std =  tf.constant(np.zeros(self.n_latent).astype(np.float32))
        
        #a_pos = tf.multiply(tf.nn.softmax(e3),M)
        #return a_pos
        return z_mu, z_log_std
    
    
    def _decoder(self, z):
        
        re_adj = tf.matmul(z, tf.transpose(z))
       
        return tf.reshape(re_adj, [-1])
    
    
    def _nb_loss(self, x, mu, sigma, eps=1e-16):
        log_mu_sigma = tf.math.log(mu + sigma + eps)
        ll = tf.math.lgamma(x + sigma) - tf.math.lgamma(sigma) - \
        tf.math.lgamma(x + 1) + sigma * tf.math.log(sigma + eps) - \
        sigma * log_mu_sigma + x * tf.math.log(mu + eps) - x * log_mu_sigma
        nb_loss = tf.reduce_sum(ll, axis=1)
        return nb_loss
        
        
    def vae_loss(self, X_in, z_pos, logits):
        ### lOSS Function
        # Multi Nominal
        #x_re_mn = tfd.Multinomial(p_re)
        #re_loss = x_mn.
        re_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, X_in))
        kl_loss = tfd.kl_divergence(z_pos, self.z_pri)
        loss = re_loss + kl_loss
        return loss
    
    
    

    
    
  
def dvae(expr, latent=5, epoch = 1000):
    ### Process
    expr = expr/np.max(expr, axis=0)
    expr = expr.T
    in_dim = expr.shape[1]
    
    
    ### Create Class
    dvae_ = BVAE(in_dim = in_dim, n_latent=latent)
    
    
    ### Input
    X_in = tf.placeholder(dtype=tf.float32, shape=[None, in_dim], name='X')

    ### Encoder
    a_pos = dvae_.encoder(X_in)
    # Create posterior distribution
    z_pos = tfd.Normal(a_pos, dvae_.var)
    
    ### Sample from Normal distribution
    z_sample = z_pos.sample()
    
    ### Decoder
    logits = dvae_.decoder(z_sample)
        
    ### lOSS Function
    re_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, X_in)
    kl_loss = tfd.kl_divergence(z_pos, dvae_.z_pri)
    #loss = re_loss - kl_loss
    elbo =tf .reduce_mean(re_loss - kl_loss)
    
    optimizer = tf.train.AdamOptimizer (0.001).minimize (-elbo)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for e in range(epoch):
        sess.run(optimizer, feed_dict = {X_in: expr})
        if not e % 50:
            elbo = sess.run(elbo, feed_dict = {X_in: expr})
            print("Epoch: %d   Loss: %f"%(e, elbo))
    
    res = sess.run(a_pos, feed_dict = {X_in: expr})
    return res