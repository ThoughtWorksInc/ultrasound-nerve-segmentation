import collections
import os

import numpy as np
import matplotlib.pyplot as plt

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

image_rows = 128
image_cols = 128


def create_train_data(input_path, output_dir):
  image_names = [
    os.path.splitext(f)[0]
    for f in os.listdir(input_path)
    if f != '.DS_Store' and 'mask' not in f
    ]

  imgs = np.ndarray((len(image_names), image_rows, image_cols, 1), dtype=np.uint8)
  img_masks = np.ndarray((len(image_names), image_rows, image_cols, 1), dtype=np.uint8)

  for i, name in enumerate(image_names):
    if i % 100 == 0:
      print('loaded: %g' % i)
    path = os.path.join(input_path, '%s.tif' % name)
    mask_path = os.path.join(input_path, '%s_mask.tif' % name)
    imgs[i] = plt.imread(path)[:, :, 1].reshape([image_rows, image_cols, 1])
    img_masks[i] = plt.imread(mask_path)[:, :, 1].reshape([image_rows, image_cols, 1])

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  np.save(os.path.join(output_dir, 'images.npy'), imgs)
  np.save(os.path.join(output_dir, 'masks.npy'), img_masks)


def create_test_data(input_path, output_dir):
  image_names = [
    os.path.splitext(f)[0]
    for f in os.listdir(input_path)
    if f != '.DS_Store' and 'mask' not in f
    ]
  imgs = np.ndarray((len(image_names), image_rows, image_cols, 1), dtype=np.uint8)

  for i, name in enumerate(image_names):
    if i % 100 == 0:
      print('loaded: %g' % i)
    path = os.path.join(input_path, '%s.tif' % name)
    imgs[i] = plt.imread(path).reshape([image_rows, image_cols, 1])

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  np.save(os.path.join(output_dir, 'images.npy'), imgs)
  np.save(os.path.join(output_dir, 'names.npy'), image_names)

def load_train_data(data_dir, train_size, test_size=None):
  raw_imgs = np.load(os.path.join(data_dir, 'images.npy'))
  raw_masks = np.load(os.path.join(data_dir, 'masks.npy'))

  num = raw_imgs.shape[0]

  assert (train_size + test_size if test_size else train_size) < num

  np.random.random_sample()

  perm = np.arange(num)
  np.random.shuffle(perm)
  train_index = perm[0:train_size]
  test_last_index = None if test_size is None else train_size + test_size
  test_index = perm[train_size:test_last_index]

  return Datasets(
    train=DataSet(raw_imgs[train_index], raw_masks[train_index]),
    test=DataSet(raw_imgs[test_index], raw_masks[test_index])
  )


def load_test_data(data_dir):
  return np.load(os.path.join(data_dir, 'names.npy')), np.load(os.path.join(data_dir, 'images.npy'))



class DataSet(object):
  def __init__(self, images, masks):
    self.images = images
    self.masks = masks

    self._num_examples = len(self.images)
    self._epochs_completed = 0
    self._index_in_epoch = 0

  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      self._epochs_completed += 1
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self.images = self.images[perm]
      self.masks = self.masks[perm]
      start = 0
      self._index_in_epoch = batch_size

    end = self._index_in_epoch

    return self.images[start:end], self.masks[start:end]


def split_image(image, output_dims, nums):
  i_height, i_width, *others = image.shape
  o_height, o_width = output_dims
  h_num, w_num = nums

  assert (i_height - o_height) % (h_num - 1) == 0
  assert (i_width - o_width) % (w_num - 1) == 0

  h_step = (i_height - o_height) // (h_num - 1)
  w_step = (i_width - o_width) // (w_num - 1)

  res = np.ndarray((h_num * w_num, o_height, o_width, *others), dtype=np.float32)
  for i in range(h_num):
    for j in range(w_num):
      res[i * w_num + j] = image[i * h_step: i * h_step + o_height, j * w_step: j * w_step + o_width]

  return res


def horizontal_merge(img1, img2, step):
  gap = img2.shape[1] - step
  return np.concatenate((img1[:, :-gap], (img1[:, -gap:] + img2[:, :gap]) / 2, img2[:, gap:]), axis=1)


def vertical_merge(img1, img2, step):
  gap = img2.shape[0] - step
  return np.concatenate((img1[:-gap, :], (img1[-gap:, :] + img2[:gap, :]) / 2, img2[gap:, :]), axis=0)


def merge(images, method, step):
  merged = np.copy(images[0])
  for img in images[1:]:
    merged = method(merged, img, step)

  return merged


def merge_image(images, output_dims, nums):
  h_num, w_num = nums
  o_height, o_width = output_dims
  i_height, i_width, *others = images[0].shape

  h_step = (o_height - i_height) // (h_num - 1)
  w_step = (o_width - i_width) // (w_num - 1)

  merged_rows = [merge(images[i * w_num: (i + 1) * w_num], method=horizontal_merge, step=w_step) for i in range(h_num)]
  merged = merge(merged_rows, method=vertical_merge, step=h_step)

  return merged