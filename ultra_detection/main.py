import os
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from ultra_detection import data
from ultra_detection.data import Datasets, DataSet
from ultra_detection.model import inference

eps = 1e-12


def dice_loss(y, y_infer):
  rounded = tf.round(y_infer)
  top = tf.reduce_sum(y * rounded, reduction_indices=[1, 2, 3])
  bottom = tf.reduce_sum(y, reduction_indices=[1, 2, 3]) + tf.reduce_sum(rounded, reduction_indices=[1, 2, 3])
  loss = tf.reduce_mean(1 - (2 * top + eps) / (bottom + eps), name='dice_loss')
  return loss, top, bottom


def training(loss):
  tf.scalar_summary(loss.op.name, loss)
  # solver
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
  return train_step


def evaluate(y, y_infer):
  eval_and = tf.logical_and(tf.greater(y, 0), tf.greater(y_infer, 0.5))
  num_intercept = tf.reduce_sum(tf.to_float(eval_and), reduction_indices=[1, 2, 3])
  num_union = tf.reduce_sum(y, reduction_indices=[1, 2, 3]) + \
              tf.reduce_sum(tf.round(y_infer), reduction_indices=[1, 2, 3])
  eval_dice = tf.reduce_mean((2 * num_intercept + eps) / (num_union + eps), name='eval_dice')
  tf.scalar_summary(eval_dice.op.name, eval_dice)
  return eval_dice, num_intercept, num_union


def run_training(datasets, log_step=10, logdir='artifacts/logs/'):
  with tf.Graph().as_default():
    # input layer
    x = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])
    y = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])

    y_infer = inference(x)
    loss, loss_top, loss_bottom = dice_loss(y, y_infer)
    train_step = training(loss)
    eval_dice, eval_intercept, eval_union = evaluate(y, y_infer)

    summary_op = tf.merge_all_summaries()

    # start session
    sess = tf.Session()

    summary_writer = tf.train.SummaryWriter(logdir, sess.graph)

    init = tf.initialize_all_variables()

    sess.run(init)

    batch_size = 50
    for i in range(100):
      batch = datasets.train.next_batch(batch_size)
      feed_dict = {x: batch[0], y: batch[1]}

      if i % log_step == 0:
        loss_res, dice_res, intercept_res, union_res, infer_res, loss_top_res, loss_bottom_res = sess.run(
          [loss, eval_dice, eval_intercept, eval_union, y_infer, loss_top, loss_bottom],
          feed_dict=feed_dict)
        print(intercept_res, union_res)

        print(loss_top_res, loss_bottom_res)

        print(infer_res.reshape(batch_size, 128 * 128))

        print("step %d, loss %g, score %g" % (i, loss_res, dice_res))
        flush_summary(summary_writer, sess, summary_op, i, feed_dict)

      sess.run(train_step, feed_dict=feed_dict)

    # evaluate test rate
    num_test = 0
    test_loss = .0
    total_dice = .0

    num_iter = len(datasets.test.images) // batch_size
    for i in range(num_iter):
      batch = datasets.test.next_batch(batch_size)
      num_test += batch_size
      y_res, batch_test_loss, dice_res = sess.run([y_infer, loss, eval_dice], feed_dict={
        x: batch[0], y: batch[1]})
      test_loss += batch_test_loss
      total_dice += dice_res
      if not os.path.exists('result'):
        os.makedirs('result')

      for j, prediction, real in zip(range(len(y_res)), y_res, batch[1]):
        plt.imsave('result/%s_%s_predict.tif' % (i, j), prediction.reshape((420, 580)), cmap='gray', vmin=0, vmax=1)
        plt.imsave('result/%s_%s_real.tif' % (i, j), real.reshape((420, 580)), cmap='gray')

    print("test total loss %g, overall score %g, test %g" % (test_loss / num_iter, total_dice / num_iter, num_test))


def flush_summary(summary_writer, sess, summary_op, i, feed_dict):
  summary_str = sess.run(summary_op, feed_dict=feed_dict)
  summary_writer.add_summary(summary_str, i)
  summary_writer.flush()


def preprocess(ultra):
  with tf.Session():
    resize_train_images = tf.image.resize_images(ultra.train.images, 128, 128)
    resize_train_masks = tf.image.resize_images(ultra.train.masks, 128, 128)
    resize_test_images = tf.image.resize_images(ultra.test.images, 128, 128)
    resize_test_masks = tf.image.resize_images(ultra.test.masks, 128, 128)

    resize_train_images = tf.cast(resize_train_images, dtype=tf.float32)
    resize_train_masks = tf.cast(resize_train_masks, dtype=tf.float32)
    resize_test_images = tf.cast(resize_test_images, dtype=tf.float32)
    resize_test_masks = tf.cast(resize_test_masks, dtype=tf.float32)

    normal_train_masks = resize_train_masks / 255.0
    normal_test_masks = resize_test_masks / 255.0

    return Datasets(
      train=DataSet(images=resize_train_images.eval(),
                    masks=normal_train_masks.eval()),
      test=DataSet(images=resize_test_images.eval(),
                   masks=normal_test_masks.eval())
    )


if __name__ == '__main__':
  data_dir = 'artifacts/data'

  if not os.path.exists(data_dir):
    data.create_train_data('../train', data_dir)

  # load data
  ultra = data.load_train_data(data_dir, 50, 10)

  processed_datasets = preprocess(ultra)
  print("train images shape: %s, test images shape: %s"
        % (processed_datasets.train.images.shape, processed_datasets.test.images.shape))

  run_training(
    processed_datasets,
    log_step=10,
    logdir=os.path.join('artifacts/logs/', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  )
