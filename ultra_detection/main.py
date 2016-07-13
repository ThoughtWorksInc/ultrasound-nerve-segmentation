import math

import tensorflow as tf

from ultra_detection.input_data import read_data_sets


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def loss(y_, y_conv):
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]), name='xentropy')
  return cross_entropy


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
      tf.truncated_normal([105 * 145 * 64, 1024], stddev=1.0 / math.sqrt(105 * 145 * 64 / 2), name='weights'))
    biases = tf.Variable(tf.constant(0.01, shape=[1024], name='biases'))

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases)

    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  with tf.name_scope('fc1'):
    # fc 2
    weights = tf.Variable(tf.truncated_normal([1024, 2], stddev=1.0 / math.sqrt(105 * 145 * 64 / 2), name='weights'))
    biases = tf.Variable(tf.constant(0.01, shape=[2], name='biases'))

    # softmax
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, weights) + biases)

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
    y_ = tf.placeholder(tf.float32, shape=[None, 2])

    # dropout prob
    keep_prob = tf.placeholder(tf.float32)

    # reshape
    x_image = tf.reshape(x, [-1, 420, 580, 1])

    y_conv = inference(keep_prob, x_image)

    # loss
    cross_entropy = loss(y_, y_conv)

    train_step = training(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    summary_op = tf.merge_all_summaries()

    # start session
    sess = tf.Session()

    summary_writer = tf.train.SummaryWriter('train_cache', sess.graph)

    init = tf.initialize_all_variables()

    sess.run(init)

    for i in range(20):
      batch = datasets.train.next_batch(50)
      if i % 5 == 0:
        acc, loss_res = sess.run([accuracy, cross_entropy], feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, loss %g, training accuracy %g" % (i, loss_res, acc))

        summary_str = sess.run(summary_op, feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

        # val_batch = datasets.validation.next_batch(10)
        # validation_accuracy = accuracy.eval(feed_dict={
        #     x: val_batch[0], y_: val_batch[1], keep_prob: 1.0
        # })
        # print("step %d, validation accuracy %g" % (i, validation_accuracy))
      sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % sess.run(accuracy, feed_dict={
      x: datasets.test.all_images, y_: datasets.test.all_labels, keep_prob: 1.0}))


# load data
ultra = read_data_sets('/Users/dtong/code/data/competition/ultrasound-nerve-segmentation/sample', 50, 10)

run_training(ultra)
