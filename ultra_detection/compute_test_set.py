import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ultra_detection.model import inference


def read_test_data(data_dir):
  return list(map(lambda p: os.path.join(data_dir, p), filter(lambda p: p != '.DS_Store' and 'mask' not in p, os.listdir(data_dir))))


def extract_img_id(path):
  return os.path.splitext(os.path.basename(path))[0]


def has_bp(cls):
  return cls[0] > cls[1]


def round_the_rect(rect, radius=10):
  if rect:
    drop_pixels = [radius - int(math.sqrt(radius ** 2 - i ** 2)) for i in range(radius, 0, -1)]
    print(rect, drop_pixels)

    for i in range(radius):
      rect[i][0] += drop_pixels[i]
      rect[i][1] -= 2 * drop_pixels[i]
      rect[-1 - i][0] += drop_pixels[i]
      rect[-1 - i][1] -= 2 * drop_pixels[i]


def generate_rle(loc):
  points = [int(loc[0] * 420), int(loc[1] * 580), int(loc[0] * 420) + int(loc[2] * 420), int(loc[1] * 580) + int(loc[3] * 580)]

  # only generate center points
  # points = [
  #   (int(loc[0] * 420) + int(loc[2] * 420)) // 2 - 3,
  #   (int(loc[1] * 580) + int(loc[3] * 580)) // 2 - 3,
  #   (int(loc[0] * 420) + int(loc[2] * 420)) // 2 + 3,
  #   (int(loc[1] * 580) + int(loc[3] * 580)) // 2 + 3]

  rect = []
  for i in range(points[1], points[3] + 1):
    rect.append([i * 420 + points[0], points[2] - points[0]])

  round_the_rect(rect, 1)

  return ' '.join(' '.join(str(item) for item in row) for row in rect)


def predict(data_paths):
  with tf.Graph().as_default():
    # input layer
    x = tf.placeholder(tf.float32, shape=[None, 420, 580])

    # dropout prob
    keep_prob = tf.placeholder(tf.float32)

    # reshape
    x_image = tf.reshape(x, [-1, 420, 580, 1])

    y_test_conv, y_test_loc = inference(keep_prob, x_image)

    init = tf.initialize_all_variables()

    sess = tf.Session()

    sess.run(init)

    saver = tf.train.Saver()

    saver.restore(sess, 'cls-model/2016-07-25_16-46-08/model.ckpt')

    print('session restored.')

    batch_size = 100
    num_data = len(data_paths)
    num_iter = num_data // batch_size + 1

    result_dir = os.path.join('predict_results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(result_dir)
    with open(os.path.join(result_dir, 'result.csv'), 'w') as f:
      for i in range(num_iter):
        end = (i + 1) * batch_size if (i + 1) * batch_size < num_data else None
        batch_paths = data_paths[i * batch_size: end]
        batch = np.array([plt.imread(p) for p in batch_paths])
        res_cls, res_loc = sess.run([y_test_conv, y_test_loc], feed_dict={x: batch, keep_prob: 1.0})

        for name, cls, loc in zip(batch_paths, res_cls, res_loc):
          print(name, loc)
          f.write('%s, %s\n' % (extract_img_id(name), generate_rle(loc)))


data_paths = read_test_data('../sample/')
predict(data_paths)
