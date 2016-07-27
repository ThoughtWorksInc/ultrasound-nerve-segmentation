import math

import tensorflow as tf


def conv(filter_size, num_filter, images):
  _, num_rows, num_cols, num_channel = images.get_shape().as_list()
  weights = tf.Variable(
    tf.truncated_normal(
      [*filter_size, num_channel, num_filter],
      stddev=1.0 / math.sqrt(num_rows * num_cols * num_channel / 2)
    ),
    name='weights'
  )
  biases = tf.Variable(tf.constant(0.01, shape=[num_filter]), name='biases')
  conv = tf.nn.relu(tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME') + biases, name='conv_relu')
  _activation_summary(conv)
  return conv


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

def inference(images):
  with tf.name_scope('conv1'):
    h_conv1 = conv([3, 3], 32, images)
    # 64 * 64
    h_pool1 = pool(h_conv1)

  with tf.name_scope('conv2'):
    h_conv2 = conv([3, 3], 64, h_pool1)
    # 32 * 32
    h_pool2 = pool(h_conv2)

  with tf.name_scope('conv3'):
    h_conv3 = conv([3, 3], 128, h_pool2)
    # 16 * 16
    h_pool3 = pool(h_conv3)

  with tf.name_scope('conv4'):
    h_conv4 = conv([3, 3], 256, h_pool3)
    # 8 * 8
    h_pool4 = pool(h_conv4)

  with tf.name_scope('conv5'):
    h_conv5 = conv([3, 3], 512, h_pool4)
    # 4 * 4
    h_pool5 = pool(h_conv5)

  with tf.name_scope('conv6'):
    h_conv6 = conv([3, 3], 1024, h_pool5)

  with tf.name_scope('upconv7'):
    # 8 * 8
    h_upsamp7 = tf.image.resize_images(h_pool5, 8, 8)
    h_upconv7 = conv([3, 3], 512, h_upsamp7)

  with tf.name_scope('merge_conv8'):
    # merge h_conv5 and h_upconv7
    h_merged8 = tf.concat(3, [h_conv5, h_upconv7])
    h_conv8 = conv([3, 3], 512, h_merged8)

  with tf.name_scope('upconv9'):
    # 16 * 16
    h_upsamp9 = tf.image.resize_images(h_conv8, 16, 16)
    h_upconv9 = conv([3, 3], 256, h_upsamp9)

  with tf.name_scope('merge_conv10'):
    h_merged10 = tf.concat(3, [h_conv4, h_upconv9])
    h_conv10 = conv([3, 3], 256, h_merged10)

  with tf.name_scope('upconv11'):
    # 32 * 32
    h_upsamp11 = tf.image.resize_images(h_conv10, 32, 32)
    h_upconv11 = conv([3, 3], 128, h_upsamp11)

  with tf.name_scope('merge_conv12'):
    h_merged12 = tf.concat(3, [h_conv3, h_upconv11])
    h_conv12 = conv([3, 3], 128, h_merged12)

  with tf.name_scope('upconv13'):
    # 64 * 64
    h_upsamp13 = tf.image.resize_images(h_conv12, 64, 64)
    h_upconv13 = conv([3, 3], 64, h_upsamp13)

  with tf.name_scope('merge_conv14'):
    h_merged14 = tf.concat(3, [h_conv2, h_upconv13])
    h_conv14 = conv([3, 3], 64, h_merged14)

  with tf.name_scope('upconv15'):
    # 64 * 64
    h_upsamp15 = tf.image.resize_images(h_conv14, 128, 128)
    h_upconv15 = conv([3, 3], 32, h_upsamp15)

  with tf.name_scope('merge_conv16'):
    h_merged16 = tf.concat(3, [h_conv1, h_upconv15])
    h_conv16 = conv([3, 3], 32, h_merged16)

  with tf.name_scope('conv17'):
    weights = tf.Variable(
      tf.truncated_normal(
        [1, 1, 32, 1],
        stddev=1.0 / math.sqrt(128 * 128 * 32)
      ),
      name='weights'
    )
    biases = tf.Variable(tf.constant(0., shape=[1]), name='biases')
    h_conv17 = tf.nn.sigmoid(tf.nn.conv2d(h_conv16, weights, strides=[1, 1, 1, 1], padding='SAME') + biases, name='conv_sigmoid')
    _activation_summary(h_conv17)
  return h_conv17