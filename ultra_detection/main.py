import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from ultra_detection import data
from ultra_detection.data import Datasets, DataSet
from ultra_detection.model import inference

eps = 1e-3


def dice_loss(y, y_infer):
  top = tf.reduce_sum(y * y_infer, reduction_indices=[1, 2, 3])
  bottom = tf.reduce_sum(y, reduction_indices=[1, 2, 3]) + tf.reduce_sum(y_infer, reduction_indices=[1, 2, 3])
  loss = tf.reduce_mean(1 - (2 * top + eps) / (bottom + eps), name='dice_loss')
  return loss, top, bottom


def training(loss):
  tf.scalar_summary(loss.op.name, loss)
  # solver
  train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
  return train_step


def evaluate(y, y_infer):
  eval_and = tf.logical_and(tf.greater(y, 0), tf.greater(y_infer, 0.5))
  num_intercept = tf.reduce_sum(tf.to_float(eval_and), reduction_indices=[1, 2, 3])
  num_union = tf.reduce_sum(y, reduction_indices=[1, 2, 3]) + \
              tf.reduce_sum(tf.round(y_infer), reduction_indices=[1, 2, 3])
  eval_dice = tf.reduce_mean((2 * num_intercept + eps) / (num_union + eps), name='eval_dice')
  tf.scalar_summary(eval_dice.op.name, eval_dice)
  return eval_dice, num_intercept, num_union


def run_training(experiment_name, datasets, log_step=10, logdir='artifacts/logs/', num_iters=500, batch_size=20):
  with tf.Graph().as_default(), tf.Session() as sess:
    # input layer
    x = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])
    y = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])

    y_infer = inference(x)
    loss, loss_top, loss_bottom = dice_loss(y, y_infer)
    train_step = training(loss)
    eval_dice, eval_intercept, eval_union = evaluate(y, y_infer)

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

    model_dir = os.path.join('artifacts/models/', experiment_name)
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    saver.save(sess, os.path.join(model_dir, 'model.ckpt'))


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

    train_mean, train_var = tf.nn.moments(resize_train_images, [0])
    test_mean, test_var = tf.nn.moments(resize_test_images, [0])
    normal_train_images = (resize_train_images - train_mean) / (tf.sqrt(train_var + eps))
    normal_test_images = (resize_test_images - test_mean) / (tf.sqrt(test_var + eps))

    normal_train_masks = resize_train_masks / 255.0
    normal_test_masks = resize_test_masks / 255.0

    return Datasets(
      train=DataSet(images=normal_train_images.eval(),
                    masks=normal_train_masks.eval()),
      test=DataSet(images=normal_test_images.eval(),
                   masks=normal_test_masks.eval())
    )


def run_testing(experiment_name, processed_datasets):
  with tf.Session() as sess:
    # real layer
    x = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])

    y_infer = inference(x, is_training=False)

    saver = tf.train.Saver()
    saver.restore(sess, os.path.join('artifacts/models', experiment_name, 'model.ckpt'))

    batch_size = 10
    num_iters = len(processed_datasets.test.images) // batch_size
    num_tested = 0
    total_dice_coef = 0.
    for i in range(num_iters):
      batch = processed_datasets.test.next_batch(batch_size)
      infer_result = sess.run(y_infer, feed_dict={x: batch[0]})
      predict = (infer_result > 0.5).astype(np.float32)
      real = batch[1]

      total_dice_coef += np.sum((2 * np.sum(real * predict, axis=(1, 2, 3)) + eps) / (
        np.sum(real, axis=(1, 2, 3)) + np.sum(predict, axis=(1, 2, 3)) + eps))

      num_tested += batch_size

    print('%g tested, average dice coef: %g' % (num_tested, total_dice_coef / num_tested))


if __name__ == '__main__':
  data_dir = 'artifacts/data'

  if not os.path.exists(data_dir):
    data.create_train_data('../train', data_dir)

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
    batch_size=20
  )

  run_testing(experiment_name, processed_datasets)
