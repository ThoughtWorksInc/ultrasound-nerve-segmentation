import math

import tensorflow as tf


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def inference(keep_prob, x_image):
  with tf.name_scope('conv1'):
    # conv 1
    weights = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=1.0 / math.sqrt(420 * 580 / 2), name='weights'))
    biases = tf.Variable(tf.constant(0.01, shape=[32], name='biases'))

    # relu + pool
    h_conv1 = tf.nn.relu(conv2d(x_image, weights) + biases)
    h_pool1 = max_pool_2x2(h_conv1)
  with tf.name_scope('conv2'):
    # conv 2
    weights = tf.Variable(
      tf.truncated_normal([5, 5, 32, 64], stddev=1.0 / math.sqrt(210 * 290 * 32 / 2), name='weights'))
    biases = tf.Variable(tf.constant(0.01, shape=[64], name='biases'))

    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights) + biases)
    h_pool2 = max_pool_2x2(h_conv2)
  h_pool2_flat = tf.reshape(h_pool2, [-1, 105 * 145 * 64])
  with tf.name_scope('fc1'):
    # fc 1
    weights = tf.Variable(
      tf.truncated_normal([105 * 145 * 64, 128], stddev=1.0 / math.sqrt(105 * 145 * 64 / 2), name='weights'))
    biases = tf.Variable(tf.constant(0.01, shape=[128], name='biases'))

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases)

    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  with tf.name_scope('cls_head'):
    # fc 2
    weights = tf.Variable(tf.truncated_normal([128, 2], stddev=1.0 / math.sqrt(105 * 145 * 64 / 2), name='weights'))
    biases = tf.Variable(tf.constant(0.01, shape=[2], name='biases'))

    # softmax
    y_cls = tf.nn.softmax(tf.matmul(h_fc1_drop, weights) + biases)
  with tf.name_scope('loc_head'):
    # fc 2
    weights = tf.Variable(tf.truncated_normal([128, 4], stddev=1.0 / math.sqrt(105 * 145 * 64 / 2), name='weights'))
    biases = tf.Variable(tf.constant(0.01, shape=[4], name='biases'))

    y_loc = tf.matmul(h_fc1_drop, weights) + biases
  return y_cls, y_loc


def loss(y_cls, y_loc, y_train_cls, y_train_loc):
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_cls * tf.log(y_train_cls), reduction_indices=[1]), name='xentropy')
  l2 = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_loc, y_train_loc), reduction_indices=[1]), name='l2_loss')
  return cross_entropy, l2


def training(cross_entropy, l2):
  tf.scalar_summary(cross_entropy.op.name, cross_entropy)
  tf.scalar_summary(l2.op.name, l2)
  # solver
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy + l2)
  return train_step