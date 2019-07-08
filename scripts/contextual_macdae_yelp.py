#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os, sys
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt 
import pandas as pd

from contextual_dataset_yelp import PretrainDatasetIter
from macDAE import RecommendMultiHeadAttnDenoisingAutoencoder

_MODEL_NAME_MACDAE="macdae"

model_head_num = None
if (len(sys.argv) == 2):
    model_head_num = sys.argv[1]

### Define input path to pretrain_model_dir
folder_path = "../"
pretrain_model_dir = None
if model_head_num is not None:
    pretrain_model_dir = os.path.join(folder_path, "model/pretrain_macdae_yelp_%s" % model_head_num)
    if not os.path.exists(pretrain_model_dir):
        print ("DEBUG: CREATING MODEL DIR: %s" % pretrain_model_dir)
        os.makedirs(pretrain_model_dir)
tf.app.flags.DEFINE_string('pretrain_model_dir', pretrain_model_dir, "model_dir")
FLAGS = tf.app.flags.FLAGS

def run_epoch(model, dataset, config):
    """ model: AutoEncoder Model
    """
    training_epochs = config.training_epochs
    batch_size = config.batch_size
    display_step = config.display_step
    model_name = config.model_name
    ## Iterate over batches
    step = 0
    costs = 0.0
    costs_reconstruct = 0.0
    costs_lagrangian = 0.0
    average_loss = 0.0
    for i, group in enumerate(dataset):
        u_idx_batch = group[0]
        u_dense_feature_batch = group[1]
        b_idx_batch = group[2]
        b_sparse_feature_batch = group[3]
        b_dense_feature_batch = group[4]
        u_i_ids_sparse = group[5]
        i_u_ids_sparse = group[6]

        param = {}
        param['user_id'] = u_idx_batch
        param['item_id'] = b_idx_batch
        param['sparse_feature'] = b_sparse_feature_batch
        param['dense_feature'] = np.concatenate([u_dense_feature_batch, b_dense_feature_batch], axis=1)
        param['u_i_ids_sparse'] = u_i_ids_sparse
        param['i_u_ids_sparse'] = i_u_ids_sparse

        cost, loss_reconstruct, loss_lagrangian = model.partial_fit(param)  # cost is total cost of batch_size example
        ## update statistics
        step += batch_size
        costs += cost
        costs_reconstruct += loss_reconstruct
        costs_lagrangian += loss_lagrangian
        average_loss = costs/step
        average_costs_reconstruct = costs_reconstruct/step
        average_costs_lagrangian = costs_lagrangian/step

        if (step % 100 == 0):
            checkpoint_path = os.path.join(FLAGS.pretrain_model_dir, model_name)
            model.save_model(checkpoint_path)
            print("Model Saved at time step %s with average reconstruct loss %f" 
                % (str(step), average_costs_reconstruct))
    return average_loss

def cluster(X, K):
    """ Cluster on the matrix X, with shape: [batch_size, n_input]
    """
    estimator = KMeans(n_clusters=K)
    estimator.fit(X)
    label_pred = estimator.labels_
    return label_pred

def visualize(X, K=3):
    """ Visualize input data X
    """
    x_input = X

    tsne=TSNE()
    tsne.fit_transform(x_input)
    labels_input = cluster(tsne.embedding_, K=K)
    tsne_df = pd.DataFrame(tsne.embedding_,index=labels_input)

    import matplotlib.pyplot as plt 
    colors = ['red', 'blue', 'green', 'burlywood','cadetblue', 'chocolate', 'cyan', 'darkgray',
        'darkorange','darkred', 'lightcoral', 'lightpink', 'lime',
        'navy','pink', 'purple', 'royalblue', 'seagreen',
        'silver','tan', 'tomato', 'violet', 'yellow',
        ]

    for k in range(K):
        d=tsne_df[labels_input==k]
        color = colors[k] if (k < len(colors)) else colors[0]
        marker = '.'
        plt.scatter(d[0],d[1], color=color, marker=marker)
    plt.show()

def eval_model(model, dataset, limit=None):
    """ classify the input tensors
    """
    batch_result = []
    # labels_result = []
    for i, group in enumerate(dataset):
        if (i % 100 == 0):
            print ("DEBUG: Processing batch number %d" % i)
        if limit is not None:
            if (i >= limit):
                break
        u_idx_batch = group[0]
        u_dense_feature_batch = group[1]
        b_idx_batch = group[2]
        b_sparse_feature_batch = group[3]
        b_dense_feature_batch = group[4]
        u_i_ids_sparse = group[5]
        i_u_ids_sparse = group[6]

        param = {}
        param['user_id'] = u_idx_batch
        param['item_id'] = b_idx_batch
        param['sparse_feature'] = b_sparse_feature_batch
        param['dense_feature'] = np.concatenate([u_dense_feature_batch, b_dense_feature_batch], axis=1)
        param['u_i_ids_sparse'] = u_i_ids_sparse
        param['i_u_ids_sparse'] = i_u_ids_sparse

        y_pred = model.transform(param)
        batch_result.append(y_pred)
        # labels_result.append(labels_list[i])
    ## merge result
    batch_array = np.concatenate(batch_result, axis=0)
    print ("DEBUG: Batch Array after concatenation is: ")
    print (batch_array.shape)
    return batch_array

class ModelConfig(object):
    ## model architect
    n_input = 403
    n_hidden = 256
    n_head = 4
    ## running hyperparameter
    learning_rate = 0.01
    dropout_probability = 0.95

class RunConfig(object):
    model_name = "auto_encoder"
    training_epochs = 5
    batch_size = 128
    display_step = 1
    examples_to_show = 10

def main():
    """ Input Pretrain dataset path: ../data/yelp/yelp-dataset/yelp_pretrain_dataset.pkl
    """
    print ("DEBUG: model_dir is:" + FLAGS.pretrain_model_dir)
    run_config = RunConfig()
    model_config = ModelConfig()
    print ("DEBUG: Input model_head_num is %d" % model_head_num) 
    model_config.n_head = model_head_num
    print ("DEBUG: Model Config n_head is %d" % model_config.n_head) 
    training_epochs = run_config.training_epochs
    data_path="../data/yelp/yelp-dataset/yelp_pretrain_dataset.pkl"
    if not os.path.exists(data_path):
        print ("DEBUG: Input Pretrain file path %s doesn't exist..." % data_path)
        return
    dataset = PretrainDatasetIter(data_path, batch_size = run_config.batch_size)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        
        # multi-head version
        model = RecommendMultiHeadAttnDenoisingAutoencoder(
            session = session,
            n_head = model_config.n_head,
            n_input=model_config.n_input,
            n_hidden=model_config.n_hidden,
            transfer_function=tf.nn.softplus,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            dropout_probability=model_config.dropout_probability,
            eps = 0.75,
            penalty_lambda=0.005)
        
        ## Train Restore model if exist
        model.restore_model(FLAGS.pretrain_model_dir)
        
        # run over epoch
        for e in range(training_epochs):
            average_loss = run_epoch(model, dataset, run_config)
            print ("DEBUG: Epoch %d average loss is %f" % (e, average_loss))
        n_limit = 100
        batch_array = eval_model(model, dataset, limit = n_limit)
        visualize(batch_array, K=8)

if __name__ == '__main__':
    if (len(sys.argv) == 2):
        model_head_num = int(sys.argv[1])
        print ("DEBUG: Input model model_head_num is %d" % model_head_num)    
    main()
