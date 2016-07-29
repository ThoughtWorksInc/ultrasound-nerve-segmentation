from unittest import TestCase

import matplotlib.pyplot as plt
from ultra_detection import compute_test

class TestComputeTest(TestCase):
  def test__should_return_rle_from_image_with_bp_nerve(self):
    image = plt.imread('ultra_detection/test/fake_data/1_1_mask.tif')
    result = compute_test.generate_rle(image)
    self.assertEqual(result, '168153 9 168570 15 168984 22 169401 26 169818 30 170236 34 170654 36 171072 39 171489 42 '
                             '171907 44 172325 46 172742 50 173159 53 173578 54 173997 55 174416 56 174834 58 '
                             '175252 60 175670 62 176088 64 176507 65 176926 66 177345 66 177764 67 178183 67 '
                             '178601 69 179020 70 179438 71 179857 71 180276 71 180694 73 181113 73 181532 73 '
                             '181945 2 181950 75 182365 79 182785 79 183205 78 183625 78 184045 77 184465 76 '
                             '184885 75 185305 75 185725 74 186145 73 186565 72 186985 71 187405 71 187825 70 '
                             '188245 69 188665 68 189085 68 189506 66 189926 65 190346 63 190766 63 191186 62 '
                             '191606 62 192026 61 192446 60 192866 59 193286 59 193706 58 194126 57 194546 56 '
                             '194966 55 195387 53 195807 53 196227 51 196647 50 197067 50 197487 48 197907 47 '
                             '198328 45 198749 42 199169 40 199589 39 200010 35 200431 33 200853 29 201274 27 '
                             '201697 20 202120 15 202544 6')

  def test_should_return_empty_string_from_image_without_bp_nerve(self):
    image = plt.imread('ultra_detection/test/fake_data/1_5_mask.tif')
    result = compute_test.generate_rle(image)
    self.assertEqual(result, '')