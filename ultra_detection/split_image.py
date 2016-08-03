import os
import sys
from ultra_detection.data import split_image
import matplotlib.pyplot as plt

input_width = 128
num_vertical = 5
num_horizontal = 5

if __name__ == '__main__':
  input_dir = sys.argv[1]
  output_dir = sys.argv[2]

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  image_names = [os.path.splitext(name)[0]
                 for name in os.listdir(input_dir)
                 if '_mask' not in name and name != '.DS_Store']

  for j, name in enumerate(image_names):
    splitted_images = split_image(
      plt.imread(os.path.join(input_dir, '%s.tif' % name)),
      (input_width, input_width),
      (num_vertical, num_horizontal)
    )
    splitted_masks = split_image(
      plt.imread(os.path.join(input_dir, '%s_mask.tif' % name)),
      (input_width, input_width),
      (num_vertical, num_horizontal)
    )

    for i in range(num_vertical * num_horizontal):
      plt.imsave(os.path.join(output_dir, '%s-%s.tif' % (name, i)), splitted_images[i], cmap='gray', vmin=0, vmax=255)
      plt.imsave(os.path.join(output_dir, '%s-%s_mask.tif' % (name, i)), splitted_masks[i], cmap='gray', vmin=0, vmax=255)

    if j % 100 == 0:
      print('finished %s' % j)
