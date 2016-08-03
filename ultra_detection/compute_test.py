import os
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ultra_detection import model, data


def generate_rle(image):
  vec = image.T.reshape((580 * 420)).tolist()
  grouped = [(value, len(list(value_iter))) for value, value_iter in groupby(vec)]
  pair_list = []
  start_index = 1
  if len(grouped) > 1:
    for ((_, num_zeros), (_, num_ones)) in zip(grouped[0::2], grouped[1::2]):
      start_index += num_zeros
      pair_list.append('%s %s' % (start_index, num_ones))
      start_index += num_ones

  return ' '.join(pair_list)



def compute_test(experiment_name, model_num, names, processed_data, threshold=0.5):
  with tf.Session() as sess:
    # real layer
    x = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])

    y_infer = model.inference(x, is_training=True)

    expanded_images = tf.image.resize_images(y_infer, 420, 580)

    saver = tf.train.Saver()
    saver.restore(sess, os.path.join('artifacts/models', experiment_name, 'model-%s.ckpt' % model_num if model_num else 'model.ckpt'))

    batch_size = 100
    num_iters = len(processed_data) // batch_size
    num_tested = 0
    for i in range(num_iters + 1):
      batch = processed_data[i * batch_size: (i + 1) * batch_size]
      images_result = sess.run(expanded_images, feed_dict={x: batch})

      result = images_result > threshold

      result_dir = 'artifacts/results/%s-%s/' % (experiment_name, model_num)
      if not os.path.exists(result_dir):
        os.makedirs(result_dir)

      num_result = images_result.shape[0]

      result = result.reshape(num_result, 420, 580)

      for j in range(num_result):
        image_index = i * batch_size + j
        plt.imsave(os.path.join(result_dir, '%s_mask.tif' % names[image_index]), result[j], cmap='gray', vmin=0, vmax=1)
        generate_rle(result[j])

      num_tested += batch_size

    print('%g tested' % num_tested)


def preprocess(images):
  eps = 1e-3
  batch_size = 2000
  num_images = len(images)
  resize_images = np.ndarray((num_images, 128, 128, 1))
  with tf.Session():
    for i in range(num_images // batch_size + 1):
      batch_images = tf.image.resize_images(images[i * batch_size: (i + 1) * batch_size], 128, 128)

      batch_images = tf.cast(batch_images, dtype=tf.float32)

      mean, var = tf.nn.moments(batch_images, [0])
      resize_images[i * batch_size: (i + 1) * batch_size] = ((batch_images - mean) / (tf.sqrt(var + eps))).eval()
  return resize_images


def generate_prediction_file(experiment_name, model_num):
  result_dir = 'artifacts/results/%s-%s/' % (experiment_name, model_num)
  image_names = [
    os.path.splitext(f)[0]
    for f in os.listdir(result_dir)
    if f != '.DS_Store'
    ]

  prediction_dir = 'artifacts/predictions/%s-%s/' % (experiment_name, model_num)
  if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)

  with open(prediction_dir + 'result.csv', 'w') as f:
    for name in image_names:
      index = name.split('_')[0]
      image = plt.imread(os.path.join(result_dir, name + '.tif'))[:, :, 1]
      rle = generate_rle(image)
      f.write('%s, %s\n' % (index, rle))



if __name__ == '__main__':
  data_dir = 'artifacts/test'
  if not os.path.exists(data_dir):
    data.create_test_data('../test', data_dir)

  names, test_data = data.load_test_data(data_dir)

  processed_data = preprocess(test_data)

  print(processed_data.shape)

  experiment_name = '2016-08-02_05-43-31'
  # compute_test(experiment_name, None, names, processed_data)
  generate_prediction_file(experiment_name, model_num=None)
