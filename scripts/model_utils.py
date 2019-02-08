#-*- coding:utf-8 -*-

import tensorflow as tf

def init_weights(shape, name = "weight"):
    """ Weight initialization, (size_A, size_B, size_C, ...)
    """
    weights = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.1),
        name=name,
        trainable=True)
    return weights

def init_weights_with_regularizer(shape, regularizer, name = "weight"):
    """ Weight initialization, (size_A, size_B, size_C, ...)
    """
    weights = tf.get_variable(name = name, initializer = tf.random_normal(shape, mean=0.0, stddev=0.1),
        regularizer = regularizer,
        trainable=True)
    return weights

def init_bias(shape, name = "bias"):
    """ shape [batch_size, output_dim]
        tf.Variable(tf.random_normal),tf.Variable
    """
    bias = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.1),
        name=name,
        trainable=True)
    return bias

def init_bias_with_regularizer(shape, regularizer, name = "bias"):
    """ shape [batch_size, output_dim]
        tf.Variable(random_normal),tf.Variable
    """
    bias = tf.get_variable(name = name, initializer = tf.random_normal(shape, mean=0.0, stddev=0.1),
        regularizer = regularizer,
        trainable=True)
    return bias
