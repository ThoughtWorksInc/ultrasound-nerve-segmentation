import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultra_detection.data import random_split_image

input_width = 320
num_splitted = 5

if __name__ == '__main__':
  input_dir = sys.argv[1]
  output_dir = sys.argv[2]

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  image_names = [os.path.splitext(name)[0]
                 for name in os.listdir(input_dir)
                 if '_mask' not in name and name != '.DS_Store']

  for j, name in enumerate(image_names):
    splitted_images, splitted_masks = random_split_image(
      plt.imread(os.path.join(input_dir, '%s.tif' % name)),
      plt.imread(os.path.join(input_dir, '%s_mask.tif' % name)),
      (input_width, input_width),
      num_splitted
    )
    for i in range(num_splitted):
      Image.fromarray(splitted_images[i]).save(os.path.join(output_dir, '%s-%s.png' % (name, i)))
      Image.fromarray(splitted_masks[i]).save(os.path.join(output_dir, '%s-%s_mask.png' % (name, i)))

      # plt.imsave(os.path.join(output_dir, '%s-%s.png' % (name, i)), splitted_images[i], cmap='gray', vmin=0, vmax=255)
      # plt.imsave(os.path.join(output_dir, '%s-%s_mask.png' % (name, i)), splitted_masks[i], cmap='gray', vmin=0, vmax=255)

    if j % 100 == 0:
      print('finished %s' % j)
