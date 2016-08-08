import math

import tensorflow as tf

from tensorflow.contrib.layers import batch_norm


def conv(filter_size, num_filter, images, is_training=True):
  _, num_rows, num_cols, num_channel = images.get_shape().as_list()
  weights = tf.Variable(
    tf.truncated_normal(
      [*filter_size, num_channel, num_filter],
      stddev=1.0 / math.sqrt(num_rows * num_cols * num_channel / 2)
    ),
    name='weights'
  )
  biases = tf.Variable(tf.constant(0.01, shape=[num_filter]), name='biases')
  conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME') + biases
  h_norm = batch_norm(conv, center=True, scale=True, is_training=is_training, scope=conv.op.name,
                      updates_collections=None)
  relu = tf.nn.elu(h_norm, name='conv_relu')
  _activation_summary(relu)
  return relu


def pool(images):
  return tf.nn.max_pool(images, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tf.histogram_summary(x.op.name + '/activations', x)
  tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


# 420 * 580 => 128 * 128

def inference(images,input_rows, input_cols, is_training=True):
  with tf.name_scope('conv1-1'):
    h_conv1_1 = conv([3, 3], 32, images, is_training=is_training)

  with tf.name_scope('conv1-2'):
    h_conv1_2 = conv([3, 3], 32, h_conv1_1, is_training=is_training)
    # 64 * 64
    h_pool1 = pool(h_conv1_2)
    input_rows //= 2
    input_cols //= 2

  with tf.name_scope('conv2-1'):
    h_conv2_1 = conv([3, 3], 64, h_pool1, is_training=is_training)

  with tf.name_scope('conv2-2'):
    h_conv2_2 = conv([3, 3], 64, h_conv2_1, is_training=is_training)
    # 32 * 32
    h_pool2 = pool(h_conv2_2)
    input_rows //= 2
    input_cols //= 2

  with tf.name_scope('conv3-1'):
    h_conv3_1 = conv([3, 3], 128, h_pool2, is_training=is_training)

  with tf.name_scope('conv3-2'):
    h_conv3_2 = conv([3, 3], 128, h_conv3_1, is_training=is_training)
    # 16 * 16
    h_pool3 = pool(h_conv3_2)
    input_rows //= 2
    input_cols //= 2

  with tf.name_scope('conv4'):
    h_conv4 = conv([3, 3], 256, h_pool3, is_training=is_training)

  with tf.name_scope('conv5'):
    h_conv5 = conv([3, 3], 512, h_conv4, is_training=is_training)

  with tf.name_scope('conv6'):
    h_conv6 = conv([3, 3], 1024, h_conv5, is_training=is_training)

  with tf.name_scope('upconv7'):
    h_upconv7 = conv([3, 3], 512, h_conv6, is_training=is_training)

  with tf.name_scope('upconv9'):
    # 16 * 16
    h_upconv9 = conv([3, 3], 256, h_upconv7, is_training=is_training)

  with tf.name_scope('merge_conv10'):
    h_merged10 = tf.concat(3, [h_conv4, h_upconv9])
    h_conv10 = conv([3, 3], 256, h_merged10, is_training=is_training)

  with tf.name_scope('upconv11'):
    # 32 * 32
    input_rows *= 2
    input_cols *= 2
    h_upsamp11 = tf.image.resize_images(h_conv10, input_rows, input_cols)
    h_upconv11 = conv([3, 3], 128, h_upsamp11, is_training=is_training)

  with tf.name_scope('merge_conv12'):
    h_merged12 = tf.concat(3, [h_conv3_2, h_upconv11])
    h_conv12 = conv([3, 3], 128, h_merged12, is_training=is_training)

  with tf.name_scope('upconv13'):
    # 64 * 64
    input_rows *= 2
    input_cols *= 2
    h_upsamp13 = tf.image.resize_images(h_conv12, input_rows, input_cols)
    h_upconv13 = conv([3, 3], 64, h_upsamp13, is_training=is_training)

  with tf.name_scope('merge_conv14'):
    h_merged14 = tf.concat(3, [h_conv2_2, h_upconv13])
    h_conv14 = conv([3, 3], 64, h_merged14, is_training=is_training)

  with tf.name_scope('upconv15'):
    # 64 * 64
    input_rows *= 2
    input_cols *= 2
    h_upsamp15 = tf.image.resize_images(h_conv14, input_rows, input_cols)
    h_upconv15 = conv([3, 3], 32, h_upsamp15, is_training=is_training)

  with tf.name_scope('merge_conv16'):
    h_merged16 = tf.concat(3, [h_conv1_2, h_upconv15])
    h_conv16 = conv([3, 3], 32, h_merged16, is_training=is_training)

  with tf.name_scope('conv17'):
    # 128 * 128
    h_upconv17 = conv([3, 3], 32, h_conv16, is_training=is_training)

  with tf.name_scope('conv18'):
    weights = tf.Variable(
      tf.truncated_normal(
        [1, 1, 32, 1],
        stddev=1.0 / math.sqrt(input_rows * input_cols * 32)
      ),
      name='weights'
    )
    biases = tf.Variable(tf.constant(0., shape=[1]), name='biases')
    conv_res = tf.nn.conv2d(h_upconv17, weights, strides=[1, 1, 1, 1], padding='SAME') + biases
    h_conv18 = tf.nn.sigmoid(conv_res, name='conv_sigmoid')
  return h_conv18
