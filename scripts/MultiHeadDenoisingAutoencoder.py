#-*- coding:utf-8 -*-

""" Implementation of the Multi-Head version of Denoise AutoEncoder(DAE) Model
"""

import tensorflow as tf

def _initialize_weights(variable_scope, n_head, n_input, n_hidden):
    """ initialize the weight of Encoder with multiple heads, and Decoder
    """
    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):       
        all_weights = dict()
        n_head_hidden = n_hidden/n_head
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
    return all_weights

def multi_head_mask_noise_autoencoder(n_head, n_input, n_hidden, transfer_function,
        x, keep_prob, variable_scope="mask_noise_autoencoder"):
    """ n_head: number of head, e.g. 8,
        n_input: dimension of input x
        n_hidden: dimension of hidden layer
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
    ## Concat the hidden layer of multiple heads
    ## output_shape:  [batch_size, n_hidden_head * n_head]
    hidden_merge = tf.concat(hidden_list, axis=1)
    print ("DEBUG: Current Hidden Merge Shape:")
    print (hidden_merge.shape)
    reconstruction = tf.add(tf.matmul(hidden_merge, network_weights['w2']), network_weights['b2'])
    return reconstruction, hidden_merge, network_weights

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

# multiple-head AutoEncoder
class RecommendMultiHeadMaskDenoisingAutoencoder(object):
    """ 
    """
    def __init__(self, session, n_head, n_input, n_hidden, 
                transfer_function = tf.nn.softplus, 
                optimizer = tf.train.AdamOptimizer(),
                dropout_probability = 0.95):
        ## input_dimension
        self.n_head = n_head
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer_function = transfer_function
        self.dropout_probability = dropout_probability
        
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
        # self.x = construct_input(self.user_id, self.item_id, self.sparse_feature, self.dense_feature, self.config)
        ## output construction 
        self.reconstruction, self.hidden_merge, self.network_weights = self.forward_pass(param)

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        
        # Optimizer
        self.optimizer = optimizer.minimize(self.cost)
        
        # Saver
        self.saver = tf.train.Saver(tf.global_variables())

        # init = tf.global_variables_initializer()
        self.sess = session
        self.sess.run(tf.global_variables_initializer())

    def partial_fit(self, param):
        """ param dict:
        """
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict = {self.keep_prob: self.dropout_probability,
                                            self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse})
        return cost

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
    
    def forward_pass(self, param, pretrain_var_scope="pretrain_model_scope"):
        """ define forward pass to hidden layer
        """
        with tf.variable_scope(pretrain_var_scope, reuse=tf.AUTO_REUSE):        
            user_id = param['user_id']
            item_id = param['item_id']
            sparse_feature = param['sparse_feature']
            dense_feature = param['dense_feature']
            u_i_ids_sparse = param['u_i_ids_sparse']
            i_u_ids_sparse = param['i_u_ids_sparse']

            self.x, self.embedding_dict = construct_input(user_id, item_id, sparse_feature, dense_feature, u_i_ids_sparse, i_u_ids_sparse, self.config)
            ## output construction
            reconstruction, hidden_merge, network_weights = multi_head_mask_noise_autoencoder(
                self.n_head, self.n_input, self.n_hidden, self.transfer_function,
                self.dropout_probability, self.x, self.keep_prob, variable_scope=pretrain_var_scope)
        return reconstruction, hidden_merge, network_weights
    
    def transform(self, param):
        """ 得到隐层
        """
        user_id = param['user_id']
        item_id = param['item_id']
        sparse_feature = param['sparse_feature']
        dense_feature = param['dense_feature']
        u_i_ids_sparse = param['u_i_ids_sparse']
        i_u_ids_sparse = param['i_u_ids_sparse']
        return self.sess.run(self.hidden_merge, feed_dict = {self.keep_prob: 1.0,
                                            self.user_id: user_id,
                                            self.item_id: item_id,
                                            self.sparse_feature: sparse_feature,
                                            self.dense_feature: dense_feature,
                                            self.u_i_ids_sparse: u_i_ids_sparse,
                                            self.i_u_ids_sparse: i_u_ids_sparse})

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
            embedding = self.sess.run(tensor,feed_dict = {self.keep_prob: 1.0,
                                            self.user_id: user_id,
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

def test():
    session = tf.Session()
    ## Pretraining first layer
    pretrain_model_config = PretrainModelConfig()
    # multi-head version of Autoencoder
    pretrain_model = MultiHeadMaskDenoisingAutoencoder(
            session = session,
            n_head = pretrain_model_config.n_head,
            n_input=pretrain_model_config.n_input,
            n_hidden=pretrain_model_config.n_hidden,
            transfer_function=tf.nn.softplus,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            dropout_probability=pretrain_model_config.dropout_probability)
    
    ## pretrain
    FLAGS = tf.app.flags.FLAGS
    print ("DEBUG: Restoreing pretrained model from dir %s" % FLAGS.pretrain_model_dir)
    pretrain_model.restore_model(FLAGS.pretrain_model_dir)
    
    ## DEBUG
    w1_1 = pretrain_model.get_weight("w1_1")
    print (w1_1)
    sum_w1_1 = np.sum(w1_1)
    print ("DEBUG: Sum of Weight %f" % sum_w1_1)

if __name__ == '__main__':
    test()
