import math
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from ultra_detection.input_data import read_data_sets


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def loss(y_, y_conv):
  cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y_, y_conv)), reduction_indices=[1]), name='xentropy')
  return cross_entropy


def inference(keep_prob, x_image):
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
    y_conv = tf.reshape(h_mask, [-1, 420*580])

  return y_conv


def training(cross_entropy):
  tf.scalar_summary(cross_entropy.op.name, cross_entropy)
  # solver
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  return train_step


def run_training(datasets):
  with tf.Graph().as_default():
    # input layer
    x = tf.placeholder(tf.float32, shape=[None, 420, 580])
    y_ = tf.placeholder(tf.float32, shape=[None, 420*580])

    # dropout prob
    keep_prob = tf.placeholder(tf.float32)

    # reshape
    x_image = tf.reshape(x, [-1, 420, 580, 1])

    y_conv = inference(keep_prob, x_image)

    # loss
    cross_entropy = loss(y_, y_conv)

    train_step = training(cross_entropy)

    summary_op = tf.merge_all_summaries()

    # start session
    sess = tf.Session()

    summary_writer = tf.train.SummaryWriter('train_cache', sess.graph)

    init = tf.initialize_all_variables()

    sess.run(init)

    batch_size = 10
    for i in range(100):
      batch = datasets.train.next_batch(batch_size)
      if i % 1 == 0:
        loss_res = sess.run(cross_entropy, feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, loss %g" % (i, loss_res))

        summary_str = sess.run(summary_op, feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

      sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # evaluate test rate
    num_test = 0
    test_loss = .0

    print(len(datasets.test.img_paths))
    for i in range(len(datasets.test.img_paths) // batch_size):
      batch = datasets.test.next_batch(batch_size)
      num_test += batch_size
      y_res, batch_test_loss = sess.run([y_conv, cross_entropy], feed_dict={
        x: batch[0], y_: batch[1], keep_prob: 1.0})
      test_loss += batch_test_loss
      if not os.path.exists('result'):
        os.makedirs('result')

      for j, prediction, real in zip(range(len(y_res)), y_res, batch[1]):
        plt.imsave('result/%s_%s_predict.tif' % (i,j), prediction.reshape((420, 580)), cmap='gray', vmin=0, vmax=1)
        plt.imsave('result/%s_%s_real.tif' % (i,j), real.reshape((420, 580)), cmap='gray')

    print("test total loss %g, test %g" % (test_loss, num_test))


# load data
ultra = read_data_sets('/Users/dtong/code/data/competition/ultrasound-nerve-segmentation/sample', 50, 10)

run_training(ultra)
