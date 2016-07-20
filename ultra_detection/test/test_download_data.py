import os
import unittest

from ultra_detection.download_data_from_s3 import download_data


class DownloadDataTestCase(unittest.TestCase):
  def test__download_files_to_path(self):
    download_data('tmp/')
    files = os.listdir('tmp')
    self.assertIn('train', files)
