import os
from datetime import datetime

import tensorflow as tf

from ultra_detection.model import inference
from ultra_detection.input_data import read_data_sets


def save_model(saver, sess, model_type):
  model_dir = os.path.join('%s-model' % model_type, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  save_path = saver.save(sess, os.path.join(model_dir, 'model.ckpt'))
  print('Model saved in file: %s' % save_path)


def loss(y_cls, y_loc, y_train_cls, y_train_loc):
  # the train rect should be within the real rect
  delta = tf.concat(1, [tf.sub(y_train_loc[:, 0:2], y_loc[:, 0:2]), tf.sub(y_loc[:, 2:], y_train_loc[:, 2:])])

  # |exp(-delta) - 1|
  return delta, tf.reduce_mean(tf.reduce_sum(tf.abs(tf.sub(tf.exp(-delta), 1)), reduction_indices=[1]), name='dice_loss')

  # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_cls * tf.log(y_train_cls), reduction_indices=[1]), name='xentropy')
  # l2 = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_loc, y_train_loc), reduction_indices=[1]), name='l2_loss')
  # return cross_entropy, l2


def training(dice_loss):
  # tf.scalar_summary(cross_entropy.op.name, cross_entropy)
  tf.scalar_summary(dice_loss.op.name, dice_loss)
  # solver
  train_step = tf.train.AdamOptimizer(1e-3).minimize(dice_loss)
  return train_step


def run_training(datasets):
  with tf.Graph().as_default():
    # input layer
    x = tf.placeholder(tf.float32, shape=[None, 420, 580])
    y_cls = tf.placeholder(tf.float32, shape=[None, 2])
    y_loc = tf.placeholder(tf.float32, shape=[None, 4])

    # dropout prob
    keep_prob = tf.placeholder(tf.float32)

    # reshape
    x_image = tf.reshape(x, [-1, 420, 580, 1])

    y_train_conv, y_train_loc = inference(keep_prob, x_image)

    # loss
    # dice_loss, l2 = loss(y_cls, y_loc, y_train_conv, y_train_loc)
    delta, dice_loss= loss(y_cls, y_loc, y_train_conv, y_train_loc)

    train_step = training(dice_loss)

    correct_prediction = tf.equal(tf.argmax(y_train_conv, 1), tf.argmax(y_cls, 1))
    eval_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    summary_op = tf.merge_all_summaries()

    # start session
    sess = tf.Session()

    summary_writer = tf.train.SummaryWriter('train_cache', sess.graph)

    init = tf.initialize_all_variables()

    sess.run(init)

    saver = tf.train.Saver()

    batch_size = 10
    for i in range(100):
      batch = datasets.train.next_batch(batch_size)
      if i % 10 == 0:
        num_correct, loss_res, loc_res, delta_res = sess.run([eval_correct, dice_loss, y_train_loc, delta], feed_dict={
          x: batch[0], y_cls: batch[1].cls, y_loc: batch[1].loc, keep_prob: 1.0})
        print("step %d, dice loss %g, training eval_correct %g" % (
          i, loss_res, num_correct / batch_size))

        summary_str = sess.run(summary_op, feed_dict={
          x: batch[0], y_cls: batch[1].cls, y_loc: batch[1].loc, keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

        print(delta_res)
        for r_loc, p_loc in zip(batch[1].loc, loc_res):
          print(r_loc, p_loc)

      sess.run(train_step, feed_dict={x: batch[0], y_cls: batch[1].cls, y_loc: batch[1].loc, keep_prob: 0.5})

    # evaluate test rate
    num_test = 0
    num_correct = 0
    num_dice = .0

    for i in range(len(datasets.test.img_paths) // batch_size):
      batch = datasets.test.next_batch(batch_size)
      num_test += batch_size
      batch_correct, batch_dice = sess.run([eval_correct, dice_loss], feed_dict={
        x: batch[0], y_cls: batch[1].cls, y_loc: batch[1].loc, keep_prob: 1.0})
      num_correct += batch_correct
      num_dice += batch_dice
    print(
      "test eval_correct %g, dice loss %g, test %g, correct %g" % (num_correct / num_test, num_dice / num_test, num_test, num_correct))

    save_model(saver, sess, 'cls')


# load data
ultra = read_data_sets('/Users/dtong/code/data/competition/ultrasound-nerve-segmentation/sample', 50, 10)

run_training(ultra)
