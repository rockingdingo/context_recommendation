#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
import numpy as np
from sklearn.metrics import roc_auc_score
from metric import ndcg_score_recommend_single

from contextual_dataset_yelp import TrainDatasetIter
from contextual_dataset_yelp import DatasetFolderIter
from contextual_dataset_yelp import TestDatasetIter
from contextual_dataset_yelp import Example, ExampleBatch
from model_utils import init_weights, init_weights_with_regularizer, init_bias, init_bias_with_regularizer

import sys
sys.path.append("../../context_recommendation/scripts/")

_MODEL_NAME_BASE = "base"
_MODEL_NAME_DAE = "dae"
_MODEL_NAME_VAE = "vae"
_MODEL_NAME_MACDAE = "macdae"

pretrain_model_name = sys.argv[1]
if pretrain_model_name.startswith(_MODEL_NAME_DAE):
    from contextual_dae_yelp import ModelConfig as DaePretrainModelConfig
    from MultiHeadDenoisingAutoencoder import RecommendMultiHeadMaskDenoisingAutoencoder
elif pretrain_model_name.startswith(_MODEL_NAME_VAE):
    from contextual_vae_yelp import ModelConfig as VaePretrainModelConfig
    from MultiHeadVariationalAutoencoder import RecommendMultiHeadVariationalAutoencoder
elif pretrain_model_name.startswith(_MODEL_NAME_MACDAE):
    from contextual_macdae_yelp import ModelConfig as MacDaePretrainModelConfig
    from macDAE import RecommendMultiHeadAttnDenoisingAutoencoder
else:
    print ("DEBUG: Input model name doesn't need Pretrain Model Config")

## deep model, middle layer relu, last layer sigmoid
def deep_part(input_x, config, regularizer = None):
    """ layers DNN, e.g. [512, 256, 256]
        input_x [batch_size, deep_feature_size]

        Args: output_layer: deep
            [batch_size, last_hidden_layer_size]
    """
    # act=tf.nn.relu
    act=tf.nn.relu
    last_act=tf.nn.sigmoid

    print ("DEBUG: Deep Part input_x shape:")
    print (tf.shape(input_x))

    output_y_dim = config.output_y_dim
    hidden_layers = config.hidden_layers
    ## Hidden Layer
    layer_num = len(hidden_layers)
    hidden = []
    for i in range(layer_num):
        input_dim = 0
        if (i == 0):
            input_dim = input_x.get_shape()[1].value   # [batch_size, concat_dim]
            print ("DEBUG: Layer %d input_dim Dim is [%d]" % (i, input_dim))
        else:
            input_dim = hidden_layers[i - 1]
        output_dim = hidden_layers[i]
        # DEBUG:
        print ("DEBUG: Layer %d Weight Dim is [%d,%d]" % (i, input_dim, output_dim))
        weight = init_weights_with_regularizer([input_dim, output_dim], name="weight_%d" % i, 
            regularizer = regularizer)
        bias = init_bias_with_regularizer([output_dim], name = "bias_%d" % i,
            regularizer = regularizer)

        if (len(hidden) == 0):
            hidden.append(act(tf.add(tf.matmul(input_x, weight), bias), name = "hidden_layer_%d" % i))
        else:
            prev_layer = hidden[len(hidden) - 1]
            if (i == (layer_num - 1)):   # lasy layer
                hidden.append(last_act(tf.add(tf.matmul(prev_layer, weight), bias), name = "hidden_layer_%d" % i))
            else:
                hidden.append(act(tf.add(tf.matmul(prev_layer, weight), bias), name = "hidden_layer_%d" % i))
    ## Output Layer
    output_layer = hidden[len(hidden) - 1]
    return output_layer, hidden

def wide_part(wide_feature, wide_dim, wide_embed_dim, name="embedding"):
    """
        wide_feature: Dense Feature
        input_wide_dim: Sparse dimension 32
        ## Results are reduced on the batch_size dimension
        Return:
            shape [batch_size, 2]
    """
    # Weight Embedding
    with tf.variable_scope("wide", reuse = tf.AUTO_REUSE):
        weight_embedding = tf.get_variable(name, 
                initializer=tf.random_normal([wide_dim, wide_embed_dim]), dtype=tf.float32)
        weight = tf.nn.embedding_lookup(weight_embedding, wide_feature,None)
        weight = tf.reduce_sum(weight, 1)
        print ("DEBUG: wide_part wide_feature shape:")
        print (wide_feature.get_shape())
        print ("DEBUG: wide_part weight shape:")
        print (weight.shape)
    return weight

def combine_deep_and_wide(wide_output, deep_output, config, regularizer = None):
    """ [batch_size, last_hidden_layer]
        wide[batch_size, 2]
    """
    output_y_dim = config.output_y_dim
    ## Merge
    deep_output_dim = deep_output.get_shape()[1].value
    ## Final Output Layer Bias
    ## Initialization
    deep_weight = init_weights_with_regularizer([deep_output_dim, output_y_dim], name="output_w",
        regularizer = regularizer)
    bias_out = init_bias_with_regularizer([output_y_dim], name="output_b", regularizer = regularizer)

    deep_final = tf.matmul(deep_output, deep_weight)
    logits = tf.add(deep_final, wide_output) + bias_out
    return logits, deep_final, wide_output, bias_out

def construct_network(model_name, pretrain_model, user_id, item_id, sparse_feature, dense_feature, u_i_ids_sparse,
        i_u_ids_sparse, config, regularizer=None):
    """ Construct Network According to model_name,
        Adding input_x or just additional features
    """
    if model_name.startswith(_MODEL_NAME_DAE):
        logits = deep_and_wide_context_pretrain(pretrain_model, user_id, item_id, sparse_feature, dense_feature, 
            u_i_ids_sparse, i_u_ids_sparse, config, regularizer)
        return logits
    elif model_name.startswith(_MODEL_NAME_VAE):
        logits = deep_and_wide_context_pretrain(pretrain_model, user_id, item_id, sparse_feature, dense_feature, 
            u_i_ids_sparse, i_u_ids_sparse, config, regularizer)
        return logits   
    elif model_name.startswith(_MODEL_NAME_MACDAE):
        logits = deep_and_wide_context_pretrain(pretrain_model, user_id, item_id, sparse_feature, dense_feature, 
            u_i_ids_sparse, i_u_ids_sparse, config, regularizer)
        return logits
    elif model_name.startswith(_MODEL_NAME_BASE):
        logits = deep_and_wide_model(user_id, item_id, sparse_feature, dense_feature, 
            u_i_ids_sparse, i_u_ids_sparse, config, regularizer)
        return logits
    else:
        return None

def deep_and_wide_model(user_id, item_id, sparse_feature, dense_feature, u_i_ids_sparse, i_u_ids_sparse, config, regularizer=None):
    """ 
    """
    output_y_dim = config.output_y_dim
    ## Deep Embedding Part
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

    ## Wide Feature Need Reduce
    wide_output = wide_part(sparse_feature, config.input_wide_dim,
            config.input_wide_embed_dim, name = "wide_output")

    print ("DEBUG: user_embed shape:")
    print (user_imp.shape)
    print ("DEBUG: item_embed shape:")
    print (item_imp.shape)
    print ("DEBUG: sparse_embed shape:")
    print (sparse_imp.shape)
    print ("DEBUG: wide_output shape:")
    print (wide_output.shape)
    # concate multiple inputs as the input to deep part
    deep_input = tf.concat([user_imp, item_imp, sparse_imp, u_i_ids_imp, i_u_ids_imp, dense_feature], 1)
    deep_output, hidden = deep_part(deep_input, config, regularizer = regularizer)
    logits, deep_final, wide_output, bias_out = combine_deep_and_wide(wide_output, deep_output, config, regularizer = regularizer)
    return logits

def deep_and_wide_context_pretrain(pretrain_model, user_id, item_id, sparse_feature, dense_feature, 
        u_i_ids_sparse, i_u_ids_sparse, config, regularizer=None):
    """ 
        Args:
            pretrain_model: pretrain_model
            user_id: Sparse shape [batch_size, 1] 
            item_id: Sparse shape [batch_size, 1] 
            sparse_feature: deep
            dense_feature:  dense
        Return:
            logits [batch_size, output_y_dim]
    """
    output_y_dim = config.output_y_dim
    ## Deep Part Using pretrained layer
    param = {}
    param['user_id'] = user_id 
    param['item_id'] = item_id
    param['sparse_feature'] = sparse_feature
    param['dense_feature'] = dense_feature
    param['u_i_ids_sparse'] = u_i_ids_sparse
    param['i_u_ids_sparse'] = i_u_ids_sparse
    
    # input_x = pretrain_model.get_input_x(param)
    pretrain_hidden_layer = _contextual_pretrain_layer(pretrain_model, param)

    ## Wide Feature Batch Dimension Reduction
    wide_output = wide_part(sparse_feature, config.input_wide_dim,
            config.input_wide_embed_dim, name = "wide_output")
    # 将 Context , deep的shop_id embedding, deep, dense CONCAT
    deep_input = pretrain_hidden_layer
    #deep_input = tf.concat([input_x, pretrain_hidden_layer], axis = 1)
    deep_output, hidden = deep_part(deep_input, config, regularizer = regularizer)
    logits, deep_final, wide_output, bias_out = combine_deep_and_wide(wide_output, deep_output, config, regularizer = regularizer)
    return logits

def _contextual_pretrain_layer(encoder_model, input_dict):
    """ 
        Args: 
            input_dict is a dict with Key: name and Value: Numpy Array
        Return: 
            output hidden layer
            forward_pass() return graph, input is tensor
    """
    reconstruction, context_hidden, _ = encoder_model.forward_pass(input_dict)
    return context_hidden

def get_fine_tune_var(exclude_variable_scope, if_freeze = True):
    """ 
        if_freeze: Boolean, if True, optimizer will exclude gradient over the
            var under variable_scope
    """
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if (if_freeze):
        final_vars = []
        exclude_vars = []
        for var in train_vars:
            if not str(var.name).startswith(exclude_variable_scope):
                final_vars.append(var)
            else:
                exclude_vars.append(var)
        ## DEBUG:
        print ("DEBUG: Excluding below Variables:")
        for var in exclude_vars:
            print (str(var.name))
        return final_vars
    else:
        return train_vars

def data_type():
    return tf.float32

class AttentionalDnnModel(object):
    """ Recommendation model with pretraining objective function
    """
    def __init__(self, pretrain_model, config):
        self._batch_size = config.batch_size
        self._config = config
        self.pretrain_model = pretrain_model
        self.pretrain_model_name = config.pretrain_model_name

        user_emb_dim = config.user_emb_dim
        item_emb_dim = config.item_emb_dim
        sparse_emb_dim = config.sparse_emb_dim
        input_dense_dim = config.input_dense_dim
        output_y_dim = config.output_y_dim

        self._user_id = tf.placeholder(tf.int32, shape=[None], name = "user_id")
        self._item_id = tf.placeholder(tf.int32, shape=[None], name = "item_id")
        self._sparse_feature = tf.placeholder(tf.int32, shape=[None, None], name = "sparse_feature")
        self._dense_feature = tf.placeholder(tf.float32, shape=[None, input_dense_dim], name = "dense_feature")
        self._u_i_ids_sparse = tf.placeholder(tf.int32, [None, None], name = "u_i_ids_sparse")
        self._i_u_ids_sparse = tf.placeholder(tf.int32, [None, None], name = "i_u_ids_sparse")

        ## pretrain_model first layer
        ## Label, shape [batch_size, 2]
        self._Y = tf.placeholder(tf.float32, shape=[None, output_y_dim], name = "Y")

        ## define regularizer
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.config.reg_scale)
        
        ## RNN
        self.logits = construct_network(self.pretrain_model_name, self.pretrain_model,
                self._user_id, self._item_id, self._sparse_feature, self._dense_feature, 
                self._u_i_ids_sparse, self._i_u_ids_sparse,
                self._config, regularizer = self.regularizer)
        
        self.prediction_result = tf.cast(tf.argmax(self.logits, 1), tf.int32, name = "output_prediction") # rename prediction to output_node for future inference
        # adding extra statistics to monitor
        self._correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), tf.cast(tf.argmax(self._Y, 1), tf.int32))
        # Click Prob
        class_prob = tf.nn.softmax(self.logits)
        ## prob for positive label
        self._pred_prob = class_prob[:,1]
        
        # Loss
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._Y, logits=self.logits))

        ## Model Training Setup
        self._lr = config.lr

        # Optimizer Partial Variable
        if config.if_freeze:
            var_to_update = get_fine_tune_var(config.varscope_pretrain, config.if_freeze)
            print ("DEBUG: Var to Update Gradient:")
            for var in var_to_update:
                print (var)
            self._train_op = tf.train.AdagradOptimizer(self._lr, 0.9).minimize(self._cost, var_list=var_to_update)
            self.saver = tf.train.Saver()
        else:
            self._train_op = tf.train.AdagradOptimizer(self._lr, 0.9).minimize(self._cost)
            self.saver = tf.train.Saver()

    @property
    def dense_feature(self):
        return self._dense_feature

    @property
    def sparse_feature(self):
        return self._sparse_feature

    @property
    def user_id(self):
        return self._user_id

    @property
    def item_id(self):
        return self._item_id

    @property
    def Y(self):
        return self._Y

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def correct_prediction(self):
        return self._correct_prediction

    @property
    def pred_prob(self):
        return self._pred_prob

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def config(self):
        return self._config

    @property
    def u_i_ids_sparse(self):
        return self._u_i_ids_sparse

    @property
    def i_u_ids_sparse(self):
        return self._i_u_ids_sparse

    @property
    def keep_prob(self):
        """ pretrain model dropout_probability placeholder
        """
        return self.pretrain_model.keep_prob

def run_epoch(sess, model, dataset, eval_op, run_config, model_name):
    """ metrics: AUC: train dataset, NDCG, test dataset
    """ 
    step = 0
    auc_score, accuracy, average_loss, costs = 0.0, 0.0, 0.0, 0.0
    y_true, y_pred = [], []
    y_true.append(np.array([1.0]))
    y_pred.append(np.array([1.0]))
    correct_examples = 0
    # for i in range(total_batch):
    for i, batch in enumerate(dataset):
        feed_dict = {}
        feed_dict[model.user_id] = batch[1]
        feed_dict[model.item_id] = batch[3]
        feed_dict[model.sparse_feature] = batch[4]
        feed_dict[model.dense_feature] = np.concatenate([batch[2], batch[5]], axis=1)
        feed_dict[model.u_i_ids_sparse] = batch[6]
        feed_dict[model.i_u_ids_sparse] = batch[7]

        batch_Y = batch[0]
        feed_dict[model.Y] = batch_Y

        if (model_name.startswith(_MODEL_NAME_DAE) or 
            model_name.startswith(_MODEL_NAME_MACDAE)):
            feed_dict[model.keep_prob] = 1.0
        ## 
        fetches = [model.cost, model.correct_prediction, model.pred_prob, eval_op]
        
        ## Feed the data
        cost, correct_prediction, pred_prob, _ = sess.run(fetches, feed_dict)
        costs += cost
        step += model.batch_size
        correct_examples += np.sum(correct_prediction)

        # AUC
        y_true.append(batch_Y[:,1])           # append positive label
        y_pred.append(pred_prob)              # append prob of positive label

        ## Save Model for each batch
        if i % run_config.interval_checkpoint_step == 10:

            # monitor statistics
            auc_score = roc_auc_score(np.concatenate(y_true, 0), np.concatenate(y_pred, 0))
            accuracy = (correct_examples/step)
            average_loss = (costs/step)

            checkpoint_path = os.path.join(tf.app.flags.FLAGS.model_dir, run_config.model_name)
            model.saver.save(sess, checkpoint_path)
            print("Model Saved at batch %d step %s with average loss %f, accuracy %f, AUC for positive label %f" 
                % (i, str(step), average_loss, accuracy, auc_score))
    return average_loss, accuracy, auc_score

def eval_epoch(sess, model, dataset, eval_op, run_config, model_name, K = [5, 10]):
    """ evaluate dataset
    """ 
    step = 0
    average_loss, costs = 0.0, 0.0
    y_true, y_pred = [], []
    # performance are average over scores
    ndcg_score_list = []
    precision_score_list = []
    recall_score_list = []

    # for i in range(total_batch):
    for i, batch in enumerate(dataset):
        feed_dict = {}
        feed_dict[model.user_id] = batch[1]
        feed_dict[model.item_id] = batch[3]
        feed_dict[model.sparse_feature] = batch[4]
        feed_dict[model.dense_feature] = np.concatenate([batch[2], batch[5]], axis=1)
        feed_dict[model.u_i_ids_sparse] = batch[6]
        feed_dict[model.i_u_ids_sparse] = batch[7]

        batch_Y = batch[0]
        feed_dict[model.Y] = batch_Y

        if (model_name.startswith(_MODEL_NAME_DAE) or 
            model_name.startswith(_MODEL_NAME_MACDAE)):
            feed_dict[model.keep_prob] = 1.0
            
        ## 
        fetches = [model.cost, model.correct_prediction, model.pred_prob, eval_op]

        ## Feed the data
        cost, correct_prediction, pred_prob, _ = sess.run(fetches, feed_dict)
        costs += cost
        step += 1

        y_true = list(batch_Y[:,1])
        y_pred = list(pred_prob)

        # Metrics: NDCG
        ndcg_level_num = len(K)
        ndcg_score = []
        for k_level in K:
            ndcg_score_k = ndcg_score_recommend_single(y_true, y_pred, k_level)
            ndcg_score.append(ndcg_score_k)

        ndcg_score_list.append(ndcg_score)
        average_loss = (costs/step)
        # monitor statistics
        if i % run_config.interval_eval_step == 10:
            # Calculate Score
            avg_ndcg_score = [0.0] * ndcg_level_num
            for j in range(ndcg_level_num):
                avg_ndcg_score[j] = np.mean([batch[j] for batch in ndcg_score_list])

            print("DEBUG: Model Evaluated at batch %d step %s with average loss %f" 
                % (i, str(step), average_loss))
            for j in range(ndcg_level_num):
                print ("DEBUG: NDCG@%d is %f" % (K[j], avg_ndcg_score[j]))

    ndcg_level_num = len(K)
    avg_ndcg_score_K_list = [0.0] * ndcg_level_num
    for j in range(ndcg_level_num):
         avg_ndcg_score_K_list[j] = np.mean([batch[j] for batch in ndcg_score_list])

    avg_ndcg_score_K = ",".join(avg_ndcg_score_K_list)
    return average_loss, _, avg_ndcg_score_K

def define_flags(model_name):
    folder_path = "../../context_recommendation"
    data_dir = os.path.join(folder_path, "data/yelp/yelp-dataset")
    model_dir = os.path.join(folder_path, "model/wide_deep_yelp_%s" % model_name)
    # Check if model_dir exists
    if not os.path.exists(model_dir):
        print ("DEBUG: CREATING TRAINING MODEL DIR: %s" % model_dir)
        os.makedirs(model_dir)
    
    pretrain_model_dir = None
    if model_name.startswith(_MODEL_NAME_MACDAE):
        multi_head_num = 4
        pretrain_model_dir = os.path.join(folder_path, "model/pretrain_macdae_yelp_%s" % multi_head_num)
    else:
        pretrain_model_dir = os.path.join(folder_path, "model/pretrain_%s_yelp" % model_name)
    tf.app.flags.DEFINE_string('data_dir', data_dir, "data_dir")
    tf.app.flags.DEFINE_string('model_dir', model_dir, "model_dir")
    # update pretrain model name
    if not model_name.startswith(_MODEL_NAME_BASE):
        tf.app.flags.FLAGS.pretrain_model_dir = pretrain_model_dir

'''
    Running Configuration
'''
class RunConfig():
    pre_activity_feature="pre_activity"
    label_column_name = "label"
    model_name = "attn_model.ckpt"
    epoch_size = 1
    interval_checkpoint_step = 500
    interval_eval_step = 1000
    interval_eval_epoch = 1

'''
    Attention Configuration
'''
class AttentionModelConfig(object):
    batch_size = 256
    # Deep Sparse Part
    user_emb_dim = 64
    item_emb_dim = 64
    sparse_emb_dim = 128
    input_dense_dim = 19
    user_size = 25000
    item_size = 170000
    sparse_feature_size = 170000
    sparse_emb_dim = 128
    input_dense_dim = 19

    total_sparse_id = 170000
    total_sparse_emb_dim = 64

    # Wide Sparse Part
    input_wide_dim = 170000
    input_wide_embed_dim = 2
    #output binary label
    output_y_dim = 2
    hidden_layers = [256]
    lr = 0.1
    max_grad_norm = 30
    reg_scale = 0.1
    
    ## namescope
    varscope_deep_sparse_feature="deep_shop_id_embedding"
    varscope_pretrain = "pretrain_model_scope"
    pretrain_model_name = "base"
    if_freeze = False

def debug_weight(pretrain_model):
    if pretrain_model is not None:
        w1_1 = pretrain_model.get_weight("w1_1")
        sum_w1_1 = np.sum(w1_1)
        print ("DEBUG: Sum of Weight w1_1 is: %f" % sum_w1_1)

def get_pretrain_model(session, model_name):
    """ Get different pretrained model...
    """
    pretrain_model = None
    if (model_name.startswith(_MODEL_NAME_DAE)):
        pretrain_model_config = DaePretrainModelConfig()
        pretrain_model = RecommendMultiHeadMaskDenoisingAutoencoder(
            session = session,
            n_head = pretrain_model_config.n_head,
            n_input=pretrain_model_config.n_input,
            n_hidden=pretrain_model_config.n_hidden,
            transfer_function=tf.nn.softplus,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            dropout_probability=pretrain_model_config.dropout_probability)
    elif (model_name.startswith(_MODEL_NAME_VAE)):
        pretrain_model_config = VaePretrainModelConfig()
        pretrain_model = RecommendMultiHeadVariationalAutoencoder(
            session = session,
            n_head=pretrain_model_config.n_head,
            n_input=pretrain_model_config.n_input,
            n_hidden=pretrain_model_config.n_hidden,
            optimizer=tf.train.AdamOptimizer(learning_rate = 0.001))
    elif (model_name.startswith(_MODEL_NAME_MACDAE)):
        pretrain_model_config = MacDaePretrainModelConfig()
        print ("DEBUG: Pretrain model_head_num is %d" % pretrain_model_config.n_head)
        pretrain_model = RecommendMultiHeadAttnDenoisingAutoencoder(
            session = session,
            n_head = pretrain_model_config.n_head,
            n_input=pretrain_model_config.n_input,
            n_hidden=pretrain_model_config.n_hidden,
            transfer_function=tf.nn.softplus,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            dropout_probability=pretrain_model_config.dropout_probability,
            eps = 0.75,
            penalty_lambda=0.005)
    elif (model_name == _MODEL_NAME_BASE):
        pretrain_model = None
    else:
        print ("DEDUG: Pretrain model name not valid %s" % model_name)
    return pretrain_model

def main(_):
    FLAGS = tf.app.flags.FLAGS
    pretrain_model_name = FLAGS.pretrain_model_name
    eval_mode = FLAGS.eval_mode
    define_flags(pretrain_model_name)
    ## Config
    # Initializing model
    run_config = RunConfig()
    model_config = AttentionModelConfig()
    model_config.pretrain_model_name = pretrain_model_name
    ## Dataset
    train_file_folder = os.path.join(FLAGS.data_dir, "train")
    test_file_folder = os.path.join(FLAGS.data_dir, "test")
    test_data_name="yelp_test_examples_1.pkl"
    test_file = os.path.join(test_file_folder, test_data_name)
    train_dataset_iter = DatasetFolderIter(train_file_folder, batch_size = model_config.batch_size)
    test_dataset_iter = TestDatasetIter(test_file)

    ## Restore Pretrain model
    with tf.Session() as session:
        pretrain_model = get_pretrain_model(session, pretrain_model_name)
        # Create New Model
        model = AttentionalDnnModel(pretrain_model, model_config)
        
        # Initialize combined model and pretrain-model
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt:
            print("Loading model parameters from %s" % FLAGS.model_dir)
            model.saver.restore(session, tf.train.latest_checkpoint(FLAGS.model_dir))
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
        debug_weight(model.pretrain_model)

        if eval_mode == "train" and pretrain_model is not None:
            print ("DEBUG: Restoreing pretrained model from dir %s" % FLAGS.pretrain_model_dir)
            pretrain_model.restore_model(FLAGS.pretrain_model_dir)
            ## DEBUG
            debug_weight(pretrain_model)

        ## train
        interval_eval_epoch = run_config.interval_eval_epoch

        ## iterate over epoch
        if eval_mode == "train":
            for e in range(run_config.epoch_size):
                # Train
                for i, train_dataset in enumerate(train_dataset_iter):
                    ## Train
                    train_loss, train_accuracy, train_auc_score = run_epoch(session, model, train_dataset, model.train_op, run_config, pretrain_model_name)
                    print("DEBUG: Epoch %d DataBatch %d: Train Session Average loss %f, accuracy %f, AUC for positive label %f" 
                        % (e, i, train_loss, train_accuracy, train_auc_score))
                    # Evaluate Pretrain Model Param
                    debug_weight(pretrain_model)
        
        elif eval_mode == "test":
            debug_weight(model.pretrain_model)
            eval_loss, _, avg_ndcg_score_K = eval_epoch(session, model, test_dataset_iter, tf.no_op(),run_config, pretrain_model_name, K = [5, 10])
            print("DEBUG: Test Session Average loss %f, NDCG for positive label %s" 
                % (eval_loss, avg_ndcg_score_K))
        else:
            print ("DEBUG: eval_mode is not applicable %s" % eval_mode)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        pretrain_model_name = sys.argv[1]
        eval_mode = sys.argv[2]
        print ("DEBUG: Runing Scripts with model name %s and mode %s" % (pretrain_model_name, eval_mode))
        tf.app.flags.DEFINE_string('pretrain_model_name', pretrain_model_name, "pretrain_model_name")
        tf.app.flags.DEFINE_string('eval_mode', eval_mode, "eval_mode")
        tf.app.run()
    else:
        print ("DEBUG: Missing input Args arg1 model_name, arg2  eval_mode")
