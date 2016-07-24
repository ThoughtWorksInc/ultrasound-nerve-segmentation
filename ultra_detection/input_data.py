import collections
import os

import numpy as np
import matplotlib.pyplot as plt

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_img(paths):
    return np.array([plt.imread(p) for p in paths], dtype=np.uint8)


def load_label(paths):
    return np.array([plt.imread(p).reshape((420*580,)) > 0 for p in paths], dtype=np.float32)


def calculate_dice(img_dir):
  paths = [os.path.join(img_dir, p) for p in os.listdir(img_dir) if p != '.DS_Store' and p.endswith('mask.tif')]
  num_images = len(paths)
  print('found %g images' % num_images)
  total_dice = .0
  for i in range(num_images):
    if i % 100 == 0:
      print('finished %g' % i)
    total_dice += 0.0000000001 / (np.sum(plt.imread(paths[i]) > 0) + 0.0000000001)
  print(total_dice / num_images)


class DataSet(object):
    def __init__(self, img_paths, label_paths):
        self.img_paths = img_paths
        self.label_paths = label_paths

        self._num_examples = len(self.img_paths)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.img_paths = self.img_paths[perm]
            self.label_paths = self.label_paths[perm]
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        return load_img(self.img_paths[start:end]), load_label(self.label_paths[start:end])

    @property
    def all_images(self):
        return load_img(self.img_paths)

    @property
    def all_labels(self):
        return load_label(self.label_paths)


def read_data_sets(train_dir, train_size, val_size, test_size=None):
    distinct_names = list(
        set(os.path.splitext(f)[0].replace('_mask', '') for f in os.listdir(train_dir) if f != '.DS_Store'))

    num = len(distinct_names)
    perm = np.arange(num)
    np.random.shuffle(perm)

    train_index = perm[0:train_size]
    val_index = perm[train_size:train_size + val_size]
    test_last_index = None if test_size is None else train_size + val_size + test_size
    test_index = perm[train_size + val_size:test_last_index]

    # TODO: concat path should use path tools
    def generate_img_paths(indexes):
        return np.array(['%s/%s.tif' % (train_dir, distinct_names[i]) for i in indexes])

    def generate_label_paths(indexes):
        return np.array(['%s/%s_mask.tif' % (train_dir, distinct_names[i]) for i in indexes])

    train_img_paths = generate_img_paths(train_index)
    train_label_paths = generate_label_paths(train_index)

    val_img_paths = generate_img_paths(val_index)
    val_label_paths = generate_label_paths(val_index)

    test_img_paths = generate_img_paths(test_index)
    test_label_paths = generate_label_paths(test_index)

    train_dataset = DataSet(train_img_paths, train_label_paths)
    val_dataset = DataSet(val_img_paths, val_label_paths)
    test_dataset = DataSet(test_img_paths, test_label_paths)

    return Datasets(train=train_dataset, validation=val_dataset, test=test_dataset)
