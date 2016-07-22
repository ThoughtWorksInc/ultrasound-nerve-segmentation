import os
import unittest

from ultra_detection.input_data import read_data_sets


def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]


class InputDataTestCase(unittest.TestCase):
    def test__read_data_sets(self):
        data = read_data_sets('ultra_detection/test/fake_data', 3, 1)

        self.assertEqual(len(data.train.img_paths), 3)
        self.assertEqual(len(data.train.label_paths), 3)
        self.assertIn(get_file_name(data.train.img_paths[0]), get_file_name(data.train.label_paths[0]))

        self.assertEqual(len(data.validation.img_paths), 1)
        self.assertEqual(len(data.validation.label_paths), 1)

        self.assertEqual(len(data.test.img_paths), 1)
        self.assertEqual(len(data.test.label_paths), 1)

    def test__next_batch_should_load_image(self):
        data = read_data_sets('ultra_detection/test/fake_data', 3, 1)

        batch = data.train.next_batch(1)

        self.assertEqual(batch[0].shape, (1, 420, 580))
        self.assertEqual(batch[1].cls.shape, (1, 2))
        self.assertEqual(batch[1].loc.shape, (1, 4))
