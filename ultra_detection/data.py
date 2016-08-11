import collections
import os

import numpy as np
import matplotlib.pyplot as plt

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

image_rows = 320
image_cols = 320
ext = 'png'


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
    path = os.path.join(input_path, '%s.%s' % (name, ext))
    mask_path = os.path.join(input_path, '%s_mask.%s' % (name, ext))
    imgs[i] = plt.imread(path)[:, :, 0].reshape([image_rows, image_cols, 1])
    img_masks[i] = plt.imread(mask_path)[:, :, 0].reshape([image_rows, image_cols, 1])

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
    path = os.path.join(input_path, '%s.%s' % (name, ext))
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

    self._positive_indexes = np.argwhere(self.masks.sum(axis=(1, 2, 3)) > 0)
    self._zero_indexes = np.argwhere(self.masks.sum(axis=(1, 2, 3)) == 0)

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

  def next_balance_batch(self, batch_size):
    half_size = batch_size // 2

    pos_choice = np.random.choice(self._positive_indexes.shape[0], half_size, replace=False)
    pos_haf_images = self.images[self._positive_indexes[pos_choice].reshape(half_size)]
    pos_haf_masks = self.masks[self._positive_indexes[pos_choice].reshape(half_size)]
    zer_choice = np.random.choice(self._zero_indexes.shape[0], half_size, replace=False)
    zer_haf_images = self.images[self._zero_indexes[zer_choice].reshape(half_size)]
    zer_haf_masks = self.masks[self._zero_indexes[zer_choice].reshape(half_size)]
    images = np.concatenate((pos_haf_images, zer_haf_images), axis=0)
    masks = np.concatenate((pos_haf_masks, zer_haf_masks), axis=0)

    return images, masks


def random_split_image(image, mask, output_dims, num):
  i_height, i_width, *others = image.shape
  o_height, o_width = output_dims

  has_bp = mask.sum() > 0

  res_images = np.ndarray((num, o_height, o_width, *others), dtype=np.uint8)
  res_masks = np.ndarray((num, o_height, o_width, *others), dtype=np.uint8)

  if has_bp:
    bp_pos = np.argwhere(mask > 0)
    top = bp_pos[:, 0].min()
    bottom = bp_pos[:, 0].max()
    left = bp_pos[:, 1].min()
    right = bp_pos[:, 1].max()
    for i in range(num):
      split_top = np.random.randint(np.maximum(bottom - o_height, 0), np.minimum(top, i_height - o_height))
      split_left = np.random.randint(np.maximum(right - o_width, 0), np.minimum(left, i_width - o_width))
      res_images[i] = image[split_top: split_top + o_height, split_left: split_left + o_width]
      res_masks[i] = mask[split_top: split_top + o_height, split_left: split_left + o_width]
  else:
    for i in range(num):
      split_top = np.random.randint(0, i_height - o_height)
      split_left = np.random.randint(0, i_width - o_width)
      res_images[i] = image[split_top: split_top + o_height, split_left: split_left + o_width]
      res_masks[i] = mask[split_top: split_top + o_height, split_left: split_left + o_width]

  return res_images, res_masks


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
