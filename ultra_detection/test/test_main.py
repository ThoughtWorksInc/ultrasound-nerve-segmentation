from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from ultra_detection.main import merge_image
from ultra_detection.data import split_image, merge_image


class MainTest(TestCase):
  def test__should_split_image(self):
    image = plt.imread('ultra_detection/test/fake_data/1_1.tif').reshape(420, 580, 1)
    res = split_image(image, (128, 128), (5, 5))
    self.assertEqual(res.shape, (25, 128, 128, 1))
    self.assertEqual(res[0, 0, 0, 0], image[0, 0, 0])
    self.assertEqual(res[1, 0, 0, 0], image[0, 113, 0])

  def test_should_merge_images(self):
    image = plt.imread('ultra_detection/test/fake_data/1_1.tif').reshape(420, 580, 1)
    res = split_image(image, (128, 128), (5, 5))
    merged = merge_image(res, (420, 580), (5, 5)).astype(np.uint8)
    self.assertTrue(np.array_equal(merged, image))
