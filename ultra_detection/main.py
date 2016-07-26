import math
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from ultra_detection import data

from ultra_detection.data import read_data_sets


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def compute_loss(y_, y_conv):
  eps = 1e-12
  top = tf.add(tf.reduce_sum(tf.mul(y_, y_conv), reduction_indices=[1]), eps)
  down = tf.add(tf.add(tf.reduce_sum(y_, reduction_indices=[1]), tf.reduce_sum(y_conv, reduction_indices=[1])), eps)
  l2_loss = tf.reduce_mean(
    -tf.div(
      top,
      down
    ),
    name='dice_loss'
  )
  return l2_loss


def training(loss):
  tf.scalar_summary(loss.op.name, loss)
  # solver
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
  return train_step


def inference(x_image):
  with tf.name_scope('conv1'):
    # conv 1
    weights = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=1.0 / math.sqrt(420 * 580 / 2), name='weights'))
    biases = tf.Variable(tf.constant(0.01, shape=[32], name='biases'))

    # relu + pool
    h_conv1 = tf.nn.relu(conv2d(x_image, weights) + biases)
    # h_pool1 = max_pool_2x2(h_conv1)
  with tf.name_scope('conv2'):
    # conv 2
    weights = tf.Variable(
      tf.truncated_normal([5, 5, 32, 64], stddev=1.0 / math.sqrt(420 * 580 * 32 / 2), name='weights'))
    biases = tf.Variable(tf.constant(0.01, shape=[64], name='biases'))

    h_conv2 = tf.nn.relu(conv2d(h_conv1, weights) + biases)
    # h_pool2 = max_pool_2x2(h_conv2)
  # h_pool2_flat = tf.reshape(h_pool2, [-1, 105 * 145 * 64])

  with tf.name_scope('conv3'):
    weights = tf.Variable(
      tf.truncated_normal([5, 5, 64, 64], stddev=1.0 / math.sqrt(420 * 580 * 64 / 2), name='weights'))
    biases = tf.Variable(tf.constant(0.01, shape=[64], name='biases'))

    h_conv3 = tf.nn.relu(conv2d(h_conv2, weights) + biases)

  with tf.name_scope('mask'):
    weights = tf.Variable(
      tf.truncated_normal([5, 5, 64, 1], stddev=1.0 / math.sqrt(420 * 580 * 64 / 2), name='weights'))
    biases = tf.Variable(tf.constant(0.01, shape=[1], name='biases'))

    h_mask = tf.nn.relu(conv2d(h_conv3, weights) + biases)
    y_conv = tf.reshape(h_mask, [-1, 420 * 580])

  return y_conv


def run_training(datasets):
  with tf.Graph().as_default():
    # input layer
    x = tf.placeholder(tf.float32, shape=[None, 420, 580])
    y_ = tf.placeholder(tf.float32, shape=[None, 420 * 580])

    # dropout prob
    keep_prob = tf.placeholder(tf.float32)

    # reshape
    x_image = tf.reshape(x, [-1, 420, 580, 1])

    y_conv = inference(x_image)

    # loss
    cross_entropy = compute_loss(y_, y_conv)

    train_step = training(cross_entropy)

    summary_op = tf.merge_all_summaries()

    eval_and = tf.logical_and(tf.greater(y_, 0), tf.greater(y_conv, 0.5))
    eval_or = tf.logical_or(tf.greater(y_, 0), tf.greater(y_conv, 0.5))
    eps = 1e-12
    eval_dice = tf.reduce_mean(
      tf.div(
        tf.add(tf.reduce_sum(tf.to_float(eval_and), reduction_indices=[1]), eps),
        tf.add(tf.reduce_sum(tf.to_float(eval_or), reduction_indices=[1]), eps)
      )
    )

    # start session
    sess = tf.Session()

    summary_writer = tf.train.SummaryWriter('train_cache', sess.graph)

    init = tf.initialize_all_variables()

    sess.run(init)

    batch_size = 50
    for i in range(20):
      batch = datasets.train.next_batch(batch_size)
      if i % 1 == 0:
        loss_res, dice_res, top_res, down_res = sess.run([cross_entropy, eval_dice], feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})

        print("step %d, loss %g, score %g" % (i, loss_res, dice_res))

        summary_str = sess.run(summary_op, feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

      sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # evaluate test rate
    num_test = 0
    test_loss = .0
    total_dice = .0

    num_iter = len(datasets.test.img_paths) // batch_size
    for i in range(num_iter):
      batch = datasets.test.next_batch(batch_size)
      num_test += batch_size
      y_res, batch_test_loss, dice_res = sess.run([y_conv, cross_entropy, eval_dice], feed_dict={
        x: batch[0], y_: batch[1], keep_prob: 1.0})
      test_loss += batch_test_loss
      total_dice += dice_res
      if not os.path.exists('result'):
        os.makedirs('result')

      for j, prediction, real in zip(range(len(y_res)), y_res, batch[1]):
        plt.imsave('result/%s_%s_predict.tif' % (i, j), prediction.reshape((420, 580)), cmap='gray', vmin=0, vmax=1)
        plt.imsave('result/%s_%s_real.tif' % (i, j), real.reshape((420, 580)), cmap='gray')

    print("test total loss %g, overall score %g, test %g" % (test_loss / num_iter, total_dice / num_iter, num_test))



if __name__ == '__main__':
  data_dir = 'artifacts/data'

  if not os.path.exists(data_dir):
    data.create_train_data('../train', data_dir)

  # load data
  # ultra = read_data_sets('/Users/dtong/code/data/competition/ultrasound-nerve-segmentation/sample', 50, 10)
  #
  # run_training(ultra)
