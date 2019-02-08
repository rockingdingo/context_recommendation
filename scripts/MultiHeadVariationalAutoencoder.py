#-*- coding:utf-8 -*-

""" Implementation of the Multi-Head version of Variational AutoEncoder(VAE) Model
"""

import tensorflow as tf

def _initialize_weights(variable_scope, n_head, n_input, n_hidden):
    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):     
        ## forward pass n_head normal random variable
        n_head_hidden = n_hidden/n_head
        all_weights = dict()
        ## The hidden size of each head
        n_hidden_head = n_hidden/n_head
        for i in range(n_head):
            index = i + 1
            ## mean vector
            mean_weight_key = 'w1_%d' % index
            mean_bias_key = 'b1_%d' % index
            ## log Variance Vector
            log_sigma_weight_key = 'log_sigma_w1_%d' % index
            log_sigma_bias_key = 'log_sigma_b1_%d' % index

            all_weights[mean_weight_key] = tf.get_variable(mean_weight_key, shape=[n_input, n_hidden_head],
                    initializer=tf.contrib.layers.xavier_initializer())
            all_weights[mean_bias_key] = tf.Variable(tf.zeros([n_hidden_head], dtype=tf.float32), name = mean_bias_key)

            all_weights[log_sigma_weight_key] = tf.get_variable(log_sigma_weight_key, shape=[n_input, n_hidden_head],
                    initializer=tf.contrib.layers.xavier_initializer())
            all_weights[log_sigma_bias_key] = tf.Variable(tf.zeros([n_hidden_head], dtype=tf.float32), log_sigma_bias_key)
        ## reconstruct process
        all_weights['w2'] = tf.Variable(tf.zeros([n_hidden, n_input], dtype=tf.float32), name = "w2")
        all_weights['b2'] = tf.Variable(tf.zeros([n_input], dtype=tf.float32), name="b2")
    return all_weights

def multi_head_variational_autoencoder(x,  n_head, n_input, n_hidden, 
        optimizer = tf.train.AdamOptimizer(),variable_scope="variational_autoencoder"):
    """ 
    """
    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):   
        network_weights = _initialize_weights(variable_scope, n_head, n_input, n_hidden)

    ## encoding
    n_hidden_head = n_hidden/n_head  # hidden dimension for each single head
    z_list = []
    z_mean_list = []
    z_log_sigma_sq_list = []
    for i in range(n_head):
        index = i + 1
        ## mean vector
        mean_weight_key = 'w1_%d' % index
        mean_bias_key = 'b1_%d' % index
        z_mean = tf.add(tf.matmul(x, network_weights[mean_weight_key]), network_weights[mean_bias_key])

        ## log variance vector
        log_sigma_weight_key = 'log_sigma_w1_%d' % index
        log_sigma_bias_key = 'log_sigma_b1_%d' % index
        z_log_sigma_sq = tf.add(tf.matmul(x, network_weights[log_sigma_weight_key]), network_weights[log_sigma_bias_key])

        # sample from gaussian distribution
        eps = tf.random_normal(tf.stack([tf.shape(x)[0], n_hidden_head]), 0, 1, dtype = tf.float32)
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
        z_list.append(z)
        z_mean_list.append(z_mean)
        z_log_sigma_sq_list.append(z_log_sigma_sq)
    ## Merging
    ## output_shape:  [batch_size, n_hidden_head * n_head]
    z_merge = tf.concat(z_list, axis=1)
    z_mean_merge = tf.concat(z_mean_list, axis = 1)
    z_log_sigma_sq_merge = tf.concat(z_log_sigma_sq_list, axis = 1)

    ## decoding
    reconstruction = tf.add(tf.matmul(z_merge, network_weights['w2']), network_weights['b2'])
    return reconstruction, z_merge, z_mean_merge, z_log_sigma_sq_merge, network_weights

"""
Recommend VAE model for Yelp Dataset
"""
class EmbeddingConfig(object):
    batch_size = 32
    total_sparse_id = 170000  # all the sparse_id total_cnt
    total_sparse_emb_dim = 64
    
    user_size = 25000
    item_size = 150000
    sparse_feature_size = 10000

    user_emb_dim = 64
    item_emb_dim = 64
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

class RecommendMultiHeadVariationalAutoencoder(object):
    """ Recommendation MultiHeadVariationalAutoencoder
    """
    def __init__(self, session, n_head, n_input, n_hidden, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_head = n_head

        # model input placeholder
        self.x = tf.placeholder(tf.float32, [None, self.n_input])

        # input placeholder
        self.config = EmbeddingConfig()

        self.user_id = tf.placeholder(tf.int32, [None], name = "user_id")
        self.item_id = tf.placeholder(tf.int32, [None], name = "item_id")
        self.sparse_feature = tf.placeholder(tf.int32, [None, None], name = "sparse_feature")
        self.dense_feature = tf.placeholder(tf.float32, [None, self.config.input_dense_dim], name = "dense_feature")
        self.u_i_ids_sparse = tf.placeholder(tf.int32, [None, None], name = "u_i_ids_sparse")
        self.i_u_ids_sparse = tf.placeholder(tf.int32, [None, None], name = "i_u_ids_sparse")

        param={}
        param['user_id'] = self.user_id
        param['item_id'] = self.item_id
        param['sparse_feature'] = self.sparse_feature
        param['dense_feature'] = self.dense_feature
        param['u_i_ids_sparse'] = self.u_i_ids_sparse
        param['i_u_ids_sparse'] = self.i_u_ids_sparse

        ## construct input
        self.reconstruction, self.z_merge, self.z_mean_merge, self.z_log_sigma_sq_merge, self.network_weights = self.forward(param)

        # cost
        reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.reconstr_loss = tf.reduce_mean(reconstr_loss)

        # print (self.z_log_sigma_sq_merge.shape)
        # print (self.z_mean_merge.shape)
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq_merge
                                           - tf.square(self.z_mean_merge)
                                           - tf.exp(self.z_log_sigma_sq_merge), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        self.optimizer = optimizer.minimize(self.cost)

        # Saver
        self.saver = tf.train.Saver(tf.global_variables())
        
        self.sess = session
        self.sess.run(tf.global_variables_initializer())

    def partial_fit(self, param):
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']

        cost, reconstr_loss,  opt = self.sess.run((self.cost, self.reconstr_loss, self.optimizer), 
                feed_dict = {self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse})
        return cost, reconstr_loss

    def forward(self, param, pretrain_var_scope="pretrain_model_scope"):
        """ forward internal use,
            Return multiple args
        """
        with tf.variable_scope(pretrain_var_scope, reuse=tf.AUTO_REUSE):
            user_id = param['user_id']
            item_id = param['item_id']
            sparse_feature = param['sparse_feature']
            dense_feature = param['dense_feature']
            u_i_ids_sparse = param['u_i_ids_sparse']
            i_u_ids_sparse = param['i_u_ids_sparse']

            # construct input tensor
            self.x, self.embedding_dict = construct_input(user_id, item_id, sparse_feature, dense_feature, u_i_ids_sparse, i_u_ids_sparse, self.config)    
            reconstruction, z_merge, z_mean_merge, z_log_sigma_sq_merge, network_weights = multi_head_variational_autoencoder(self.x, self.n_head, 
                self.n_input, self.n_hidden, optimizer = tf.train.AdamOptimizer(), variable_scope=pretrain_var_scope)
        return reconstruction, z_merge, z_mean_merge, z_log_sigma_sq_merge, network_weights

    def forward_pass(self, param, pretrain_var_scope="pretrain_model_scope"):
        """ external use, use z_mean_merge(without the normal component) 
            as the forward feature extraction, not the z_merge
        """
        reconstruction, z_merge, z_mean_merge, z_log_sigma_sq_merge, network_weights = self.forward(param, pretrain_var_scope)
        return reconstruction, z_mean_merge, network_weights
    
    def calc_total_cost(self, param):
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']
        return self.sess.run(self.cost, feed_dict = {self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse})

    def calc_reconstruct_cost(self, param):
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']
        return self.sess.run(self.reconstr_loss, feed_dict = {self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse})
    
    def transform(self, param):
        """ transform input_x to z_merge vector
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
    
    def transform_z_mean(self, param):
        """ transform input_x to z_mean_merge vector
        """
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']        
        return self.sess.run(self.z_mean_merge, feed_dict = {self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse})

    def save_model(self, checkpoint_path):
        self.saver.save(self.sess, checkpoint_path)
        return 

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

    def restore_model(self, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            print("Loading model parameters from %s" % model_dir)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
        return

