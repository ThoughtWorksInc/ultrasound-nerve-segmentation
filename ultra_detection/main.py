import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from ultra_detection import data
from ultra_detection.data import Datasets, DataSet
from ultra_detection.model import inference


def dice_loss(y, y_infer):
  eps = 1
  top = tf.reduce_sum(y * y_infer, reduction_indices=[1, 2, 3])
  bottom = tf.reduce_sum(y, reduction_indices=[1, 2, 3]) + tf.reduce_sum(y_infer, reduction_indices=[1, 2, 3])
  loss = tf.reduce_mean(1 - (2 * top + eps) / (bottom + eps), name='dice_loss')
  return loss


def l2_loss(y, y_infer):
  return tf.nn.l2_loss(y_infer - y)


def training(loss):
  tf.scalar_summary(loss.op.name, loss)
  # solver
  train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
  return train_step


def evaluate(y, y_infer, threshold):
  eps = 1e-3
  eval_and = tf.logical_and(tf.greater(y, 0), tf.greater(y_infer, threshold))
  num_intercept = tf.reduce_sum(tf.to_float(eval_and), reduction_indices=[1, 2, 3])
  num_union = tf.reduce_sum(y, reduction_indices=[1, 2, 3]) + \
              tf.reduce_sum(tf.round(y_infer), reduction_indices=[1, 2, 3])
  eval_dice = tf.reduce_mean((2 * num_intercept + eps) / (num_union + eps), name='eval_dice')
  tf.scalar_summary(eval_dice.op.name, eval_dice)
  return eval_dice


def run_training(experiment_name,
                 datasets, log_step=10,
                 logdir='artifacts/logs/',
                 num_iters=500,
                 batch_size=20,
                 check_step=100):
  model_dir = os.path.join('artifacts/models/', experiment_name)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  with tf.Graph().as_default(), tf.Session() as sess:
    # input layer
    x = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])
    y = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])

    y_infer = inference(x)
    loss = l2_loss(y, y_infer)
    train_step = training(loss)
    eval_dice = evaluate(y, y_infer, 0.5)

    tf.image_summary('train_masks', y_infer, max_images=20)
    tf.image_summary('real_masks', y, max_images=20)

    summary_op = tf.merge_all_summaries()

    if not os.path.exists(logdir):
      os.makedirs(logdir)

    summary_writer = tf.train.SummaryWriter(os.path.join(logdir, experiment_name), sess.graph)

    init = tf.initialize_all_variables()

    sess.run(init)

    saver = tf.train.Saver()

    for i in range(num_iters):
      batch = datasets.train.next_batch(batch_size)
      feed_dict = {x: batch[0], y: batch[1]}

      if i % log_step == 0 or i == num_iters - 1:
        flush_summary(summary_writer, sess, summary_op, i, feed_dict)

        loss_res, dice_res, infer_res = sess.run([loss, eval_dice, y_infer], feed_dict=feed_dict)

        num_val_iter = len(datasets.test.images) // batch_size
        total_dice = .0

        for j in range(num_val_iter):
          test_batch = datasets.test.next_batch(batch_size)
          test_batch_dice_0_5 = sess.run(eval_dice, feed_dict={x: test_batch[0], y: test_batch[1]})
          total_dice += test_batch_dice_0_5
        total_dice /= num_val_iter

        print("step %d, loss %g, 0.5 score %g, 0.5 validation dice: %g" %
              (i, loss_res, dice_res, total_dice))

      if i % check_step == 0 or i == num_iters - 1:
        saver.save(sess, os.path.join(model_dir, 'model-%g.ckpt' % i))

      sess.run(train_step, feed_dict=feed_dict)

    saver.save(sess, os.path.join(model_dir, 'model.ckpt'))


def flush_summary(summary_writer, sess, summary_op, i, feed_dict):
  summary_str = sess.run(summary_op, feed_dict=feed_dict)
  summary_writer.add_summary(summary_str, i)
  summary_writer.flush()


def preprocess(ultra):
  ultra.train.images = ultra.train.images.astype(np.float32)
  ultra.train.masks = ultra.train.masks.astype(np.float32)
  ultra.test.images = ultra.test.images.astype(np.float32)
  ultra.test.masks = ultra.test.masks.astype(np.float32)

  ultra.train.images -= np.mean(ultra.train.images)
  ultra.train.masks -= np.mean(ultra.train.masks)
  ultra.train.images /= np.linalg.norm(ultra.train.images)
  ultra.train.masks /= np.linalg.norm(ultra.train.masks)

  ultra.test.images -= np.mean(ultra.test.images)
  ultra.test.masks -= np.mean(ultra.test.masks)

  ultra.test.images /= np.linalg.norm(ultra.test.images)
  ultra.test.masks /= np.linalg.norm(ultra.test.masks)

  return Datasets(
    train=DataSet(images=ultra.train.images, masks=ultra.train.masks),
    test=DataSet(images=ultra.test.images, masks=ultra.test.masks)
  )


if __name__ == '__main__':
  data_dir = 'artifacts/splitted_data'

  if not os.path.exists(data_dir):
    data.create_train_data('../splitted_train', data_dir)

  # load data
  ultra = data.load_train_data(data_dir, 20, 10)

  processed_datasets = preprocess(ultra)
  print("train images shape: %s, test images shape: %s"
        % (processed_datasets.train.images.shape, processed_datasets.test.images.shape))

  experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  run_training(
    experiment_name,
    processed_datasets,
    log_step=10,
    logdir='artifacts/logs/',
    num_iters=100,
    batch_size=20,
    check_step=100
  )
