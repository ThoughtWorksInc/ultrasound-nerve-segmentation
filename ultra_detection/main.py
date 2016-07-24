import os
from datetime import datetime

import tensorflow as tf

from ultra_detection.model import inference, loss, training
from ultra_detection.input_data import read_data_sets


def save_model(saver, sess):
  model_dir = os.path.join('model', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  save_path = saver.save(sess, os.path.join(model_dir, 'model.ckpt'))
  print('Model saved in file: %s' % save_path)


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
    cross_entropy, l2 = loss(y_cls, y_loc, y_train_conv, y_train_loc)

    train_step = training(cross_entropy, l2)

    correct_prediction = tf.equal(tf.argmax(y_train_conv, 1), tf.argmax(y_cls, 1))
    eval_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    summary_op = tf.merge_all_summaries()

    # start session
    sess = tf.Session()

    summary_writer = tf.train.SummaryWriter('train_cache', sess.graph)

    init = tf.initialize_all_variables()

    sess.run(init)

    saver = tf.train.Saver()

    batch_size = 50
    for i in range(100):
      batch = datasets.train.next_batch(batch_size)
      if i % 10 == 0:
        num_correct, loss_res, l2_res = sess.run([eval_correct, cross_entropy, l2], feed_dict={
          x: batch[0], y_cls: batch[1].cls, y_loc: batch[1].loc, keep_prob: 1.0})
        print("step %d, cls loss %g, loc_loss %g, training eval_correct %g" % (
          i, loss_res, l2_res, num_correct / batch_size))

        summary_str = sess.run(summary_op, feed_dict={
          x: batch[0], y_cls: batch[1].cls, y_loc: batch[1].loc, keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

      sess.run(train_step, feed_dict={x: batch[0], y_cls: batch[1].cls, y_loc: batch[1].loc, keep_prob: 0.5})

    # evaluate test rate
    num_test = 0
    num_correct = 0
    num_l2 = .0

    for i in range(len(datasets.test.img_paths) // batch_size):
      batch = datasets.test.next_batch(batch_size)
      num_test += batch_size
      batch_correct, batch_l2 = sess.run([eval_correct, l2], feed_dict={
        x: batch[0], y_cls: batch[1].cls, y_loc: batch[1].loc, keep_prob: 1.0})
      num_correct += batch_correct
      num_l2 += batch_l2

    print(
      "test eval_correct %g, l2 loss %g, test %g, correct %g" % (num_correct / num_test, num_l2, num_test, num_correct))

    save_model(saver, sess)


# load data
ultra = read_data_sets('/Users/dtong/code/data/competition/ultrasound-nerve-segmentation/sample', 50, 10)

run_training(ultra)
