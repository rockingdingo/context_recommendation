#-*- coding:utf-8 -*-

""" Implementation of the Mixture Attentional Denoise AutoEncoder(MACDAE) Model
"""

import numpy as np
import tensorflow as tf
from model_utils import init_weights, init_bias
import scipy.spatial.distance as distance
from scipy import linalg, mat, dot

def _initialize_weights(variable_scope, n_head, n_input, n_hidden):
    """ initialize the weight of Encoder with multiple heads, and Decoder
    """
    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):       
        all_weights = dict()
        ## forward, Each head has the same dimension ad n_hidden
        n_head_hidden = n_hidden/n_head
        # n_head_hidden = n_hidden
        for i in range(n_head):
            index = i + 1
            weight_key = 'w1_%d' % index
            bias_key = 'b1_%d' % index
            all_weights[weight_key] = tf.get_variable(weight_key, shape=[n_input, n_head_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
            all_weights[bias_key] = tf.Variable(tf.zeros([n_head_hidden], dtype = tf.float32), name = bias_key)
        ## reconstruct 
        all_weights['w2'] = tf.Variable(tf.zeros([n_hidden, n_input], dtype = tf.float32), name = 'w2')
        all_weights['b2'] = tf.Variable(tf.zeros([n_input], dtype = tf.float32), name = 'b2')
    ## DEBUG:
    for key in all_weights.keys():
        tensor = all_weights[key]
        print ("DEBUG: Shape of Weight %s" % key)
        print (tensor.shape)
    return all_weights

def multi_head_attn_autoencoder(n_head, n_input, n_hidden, transfer_function,
        dropout_probability, x, keep_prob, variable_scope="mask_noise_autoencoder"):
    """ mixture of multi-head Attentional Constrained Model
        attn_weight denotes the attentional weight input x place on the k-th hidden state, implicit contexts
    """
    network_weights = _initialize_weights(variable_scope, n_head, n_input, n_hidden)
    # model
    hidden_list = []
    for i in range(n_head):
        index = i + 1
        weight_key_i = 'w1_%d' % index
        bias_key_i = 'b1_%d' % index
        hidden = transfer_function(tf.add(tf.matmul(tf.nn.dropout(x, keep_prob), 
                    network_weights[weight_key_i]),network_weights[bias_key_i]))
        hidden_list.append(hidden)
    # 
    head_dim = n_hidden/n_head
    # head_dim = n_hidden
    W = init_weights([n_input, head_dim], name="W")
    x_projection = tf.matmul(x, W)
    dot_prod_list = []
    for i in range(n_head):
        hidden = hidden_list[i]
        hidden_dot = tf.reduce_sum(tf.multiply(x_projection, hidden), axis=1)
        dot_prod_list.append(hidden_dot)
    dot_prod = tf.stack(dot_prod_list, axis = 1)
    attn_weight = tf.nn.softmax(dot_prod, axis = 1)
    # Get Hidden representation
    hidden_merge_list = []
    for i in range(n_head):
        weight_hidden = tf.multiply(tf.stack([attn_weight[:,i]]*head_dim, axis=1), hidden_list[i])
        hidden_merge_list.append(weight_hidden)
    # hidden_merge = tf.reduce_sum(tf.stack(hidden_merge_list, axis=1), axis=1)
    hidden_merge = tf.concat(hidden_merge_list, axis = 1)
    print ("DEBUG: Current Hidden Merge Shape:")
    print (hidden_merge.shape)
    reconstruction = tf.add(tf.matmul(hidden_merge, network_weights['w2']), network_weights['b2'])
    return reconstruction, hidden_merge, hidden_list, network_weights, attn_weight

def lagrangian_penalty_loss(z_list, eps, penalty_lambda, M = 1.0):
    """ eps constraint threshold
    """ 
    n_head = len(z_list)
    batch_size=z_list[0].shape[0].value
    hidden_dim=z_list[0].shape[1].value
    prod_list = []
    lambda_list = []
    cosine_list = []
    for i in range(n_head):
        for j in range(i+1, n_head):
            z_i = z_list[i]
            z_j = z_list[j]
            ##eucli_dist_batch = tf.expand_dims(tf.reduce_sum(tf.square(z_i-z_j), axis = 1), 1)
            cos_sim_batch = get_cosine_similarity(z_i, z_j)
            cosine_list.append(cos_sim_batch)
            slack_batch = cos_sim_batch - eps
            prod = tf.multiply(slack_batch, penalty_lambda)
            prod_list.append(prod)
    ## sum
    lagrangian_loss = 0.0
    print ("DEBUG: prod_list size is %d" % len(prod_list))
    if len(prod_list) == 0:
        lagrangian_loss = tf.Variable(0.0)
    else:
        lagrangian_loss = 0.5 * tf.reduce_mean(tf.concat(prod_list, axis = 0))  # [?,] merge[?,]
        lagrangian_loss = lagrangian_loss/M
    return lagrangian_loss, cosine_list, lambda_list

def get_cosine_similarity(x, y):
    """ x  [batch_size, dim]
        y  [batch_size, dim]
        Return : cos_similarity  1.0 for 0 degree, -1 for 180 degreee
    """
    norm_x=tf.nn.l2_normalize(x, 1)
    norm_y=tf.nn.l2_normalize(y, 1)
    cos_similarity=tf.reduce_sum(tf.multiply(norm_x,norm_y), axis=1)
    return cos_similarity

def get_cosine_similarity_numpy(x, y):
    cos_similarity = 1 - distance.cosine(x, y)
    return cos_similarity

def get_cosine_similarity_mat(x, y):
    cos = dot(x,y.T)/linalg.norm(x)/linalg.norm(y)
    return cos

# x= np.array([[1.0,2.0,3.0], [1.0,1.0,1.0]])
# y= np.array([[1.0,2.0,3.0], [1.0,1.0,2.0]])
# cos_sim = get_cosine_similarity(x,y)
# cos_sim_np = get_cosine_similarity_numpy(x,y)
# cos_sim_mat = get_cosine_similarity_mat(x,y)

class EmbeddingConfig(object):
    batch_size = 32
    # all the sparse_id total_cnt
    total_sparse_id = 170000
    total_sparse_emb_dim = 64
    
    user_size = 25000
    user_emb_dim = 64
    item_size = 150000
    item_emb_dim = 64
    sparse_feature_size = 10000
    sparse_emb_dim = 128
    input_dense_dim = 19

def data_type():
    return tf.float32

def construct_input(user_id, item_id, sparse_feature, dense_feature, u_i_ids_sparse, i_u_ids_sparse, config):
    """ construct input tensor
        user_id:  [batch_size]
        item_id:  [batch_size]
        sparse_feature: shape [None, None]
    """
    embedding_dict = {}
    with tf.variable_scope('embedding',reuse=tf.AUTO_REUSE):
        sparse_embedding = tf.get_variable("sparse_embedding", [config.total_sparse_id, config.total_sparse_emb_dim], dtype=data_type())
        # User
        user_imp = tf.nn.embedding_lookup(sparse_embedding, user_id)
        # Item
        item_imp = tf.nn.embedding_lookup(sparse_embedding, item_id)
        # Sparse
        sparse_feature_imp = tf.nn.embedding_lookup(sparse_embedding, sparse_feature)
        sparse_imp = tf.reduce_mean(sparse_feature_imp, axis=1)
        # U-I Embedding
        u_i_ids_feature_imp = tf.nn.embedding_lookup(sparse_embedding, u_i_ids_sparse)
        u_i_ids_imp = tf.reduce_mean(u_i_ids_feature_imp, axis=1)
        # I-U Embedding
        i_u_ids_feature_imp = tf.nn.embedding_lookup(sparse_embedding, i_u_ids_sparse)
        i_u_ids_imp = tf.reduce_mean(i_u_ids_feature_imp, axis=1)
    # Two-Way interaction
    prod_vector = 0.5 * (1.0/10.0) * (tf.multiply(user_imp, item_imp)
        + tf.multiply(user_imp, sparse_imp)
        + tf.multiply(user_imp, u_i_ids_imp)
        + tf.multiply(user_imp, i_u_ids_imp)
        + tf.multiply(item_imp, sparse_imp)
        + tf.multiply(item_imp, u_i_ids_imp)
        + tf.multiply(item_imp, i_u_ids_imp)
        + tf.multiply(sparse_imp, u_i_ids_imp)
        + tf.multiply(sparse_imp, i_u_ids_imp)
        + tf.multiply(u_i_ids_imp, i_u_ids_imp))
    # Two-Way interaction
    # prod_vector = tf.multiply(user_imp, item_imp)
    imp = tf.concat([user_imp, item_imp, sparse_imp, u_i_ids_imp, i_u_ids_imp, prod_vector, dense_feature], axis = 1)
    embedding_dict["user_embed"] = user_imp
    embedding_dict["item_embed"] = item_imp
    embedding_dict["input_x"] = imp        
    return imp, embedding_dict

class RecommendMultiHeadAttnDenoisingAutoencoder(object):
    """ Denoising autoencoder
    """
    def __init__(self, session, n_head, n_input, n_hidden, 
                transfer_function = tf.nn.softplus, 
                optimizer = tf.train.AdamOptimizer(),
                dropout_probability = 0.95,
                eps = 0.9,
                penalty_lambda = 0.005,
                if_penalty = True):
        ## input_dimension
        self.n_head = n_head
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer_function = transfer_function
        self.dropout_probability = dropout_probability

        # input placeholder
        self.config = EmbeddingConfig()

        # input placeholder
        self.user_id = tf.placeholder(tf.int32, [None], name = "user_id")
        self.item_id = tf.placeholder(tf.int32, [None], name = "item_id")
        self.sparse_feature = tf.placeholder(tf.int32, [None, None], name = "sparse_feature")
        self.dense_feature = tf.placeholder(tf.float32, [None, self.config.input_dense_dim], name = "dense_feature")
        self.u_i_ids_sparse = tf.placeholder(tf.int32, [None, None], name = "u_i_ids_sparse")
        self.i_u_ids_sparse = tf.placeholder(tf.int32, [None, None], name = "i_u_ids_sparse")

        self.keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

        param={}
        param['user_id'] = self.user_id
        param['item_id'] = self.item_id
        param['sparse_feature'] = self.sparse_feature
        param['dense_feature'] = self.dense_feature
        param['u_i_ids_sparse'] = self.u_i_ids_sparse
        param['i_u_ids_sparse'] = self.i_u_ids_sparse

        ## output reconstruction
        self._reconstruction, self._hidden_merge, self._hidden_list, self.network_weights, self.attn_weight = self.forward(param)

        # reconstruction loss
        self.loss_reconstruct = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self._reconstruction, self.x), 2.0))
        self.loss_lagrangian, self.cosine_list, self.lambda_list = lagrangian_penalty_loss(self._hidden_list, eps, penalty_lambda = penalty_lambda, M = 1)
        if if_penalty:
            self.cost = self.loss_reconstruct + self.loss_lagrangian
        else:
            self.cost = self.loss_reconstruct

        # Optimizer
        self.optimizer = optimizer.minimize(self.cost)

        # Saver
        self.saver = tf.train.Saver(tf.global_variables())

        # init = tf.global_variables_initializer()
        self.sess = session
        self.sess.run(tf.global_variables_initializer())
    
    def partial_fit(self, param):
        """ train epoch, run partial_fit and optimizer
        """
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']

        # fetch_list = [self.cost, self.loss_reconstruct, self.loss_lagrangian, self.optimizer]
        fetch_list = [self.cost, self.loss_reconstruct, self.loss_lagrangian, self.optimizer]
        cost, loss_reconstruct, loss_lagrangian, _ = self.sess.run(fetch_list,
                                  feed_dict = {self.keep_prob: self.dropout_probability,
                                            self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse})
        return cost, loss_reconstruct, loss_lagrangian

    def calc_total_cost(self, param):
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']         
        return self.sess.run(self.cost, feed_dict = {self.keep_prob: 1.0,
                                            self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse})
        
    def forward(self, param, pretrain_var_scope="pretrain_model_scope"):
        """ internal use of forward network, forward pass and Return
            multiple results
        """
        with tf.variable_scope(pretrain_var_scope, reuse=tf.AUTO_REUSE):
            user_id = param['user_id']
            item_id = param['item_id']
            sparse_feature = param['sparse_feature']
            dense_feature = param['dense_feature']
            u_i_ids_sparse = param['u_i_ids_sparse']
            i_u_ids_sparse = param['i_u_ids_sparse'] 

            self.x, self.embedding_dict = construct_input(user_id, item_id, sparse_feature, dense_feature, 
                u_i_ids_sparse, i_u_ids_sparse, self.config)
            reconstruction, hidden_merge, hidden_list, network_weights, attn_weight = multi_head_attn_autoencoder(
                self.n_head, self.n_input, self.n_hidden, self.transfer_function,
                self.dropout_probability, self.x, self.keep_prob, variable_scope=pretrain_var_scope)
        return reconstruction, hidden_merge, hidden_list, network_weights, attn_weight

    def forward_pass(self, param, pretrain_var_scope="pretrain_model_scope"):
        """ define forward pass to hidden layer, 
            return hidden layer
        """
        reconstruction, hidden_merge, hidden_list, network_weights, attn_weight = self.forward(param, pretrain_var_scope)
        return reconstruction, hidden_merge, network_weights
    
    def get_input_x(self, param):
        """ Get the constructed input from param
        """
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']
        x = construct_input(user_id, item_id, sparse_feature, dense_feature, u_i_ids_sparse, i_u_ids_sparse, self.config)
        return x
        
    def transform(self, param):
        """ get the hidden merge layer of the network
        """
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']

        return self.sess.run(self._hidden_merge, feed_dict = {self.keep_prob: 1.0,
                                            self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse
                                            })

    def transform_z_mean(self, param):
        """ transform input x to z_mean
        """
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']          
        return self.sess.run(self.z_merge, feed_dict = {self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse})

    def calc_attention(self, param):
        """ input keep prob and get attentional on different weights
        """
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']
        return self.sess.run(self.attn_weight, feed_dict = {self.keep_prob: 0.8,
                                            self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse
                                            })

    def get_weight(self, name):
        """ get the weight under name
        """
        tensor = None
        if name in self.network_weights.keys():
            tensor = self.network_weights[name]
            weight = self.sess.run(tensor)
            return weight
        else:
            return None

    def get_embedding(self, param, name):
        """ get the Embedding vector of user, item
        """
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']        
        tensor = None
        if name in self.embedding_dict.keys():
            tensor = self.embedding_dict[name]
            embedding = self.sess.run(tensor,feed_dict = {self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse})
            return embedding
        else:
            return None
    
    def save_model(self, checkpoint_path):
        self.saver.save(self.sess, checkpoint_path)
        return 
    
    def restore_model(self, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            print("Loading model parameters from %s" % model_dir)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
        return

    @property
    def hidden_list(self):
        return self._hidden_list

    @property
    def reconstruction(self):
        return self._reconstruction
