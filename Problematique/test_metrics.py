import unittest

from triton.language import tensor

from metrics import *

class TestMetrics(unittest.TestCase):

    def test_metrics_mot_pareil(self):
        mot_a = 'mot'
        mot_b = 'mot'

        self.assertEqual(edit_distance(mot_a, mot_b), 0)

    def test_metrics_mot_diff(self):
        mot_a = 'chien'
        mot_b = 'niche'

        self.assertEqual(edit_distance(mot_a, mot_b), 4)

    def test_confusion_matrix(self):
        true = [2, 0, 2, 2, 0, 1]
        pred = [0, 0, 2, 2, 0, 2]

        array_matrix = confusion_matrix_update(true, pred)

        # Flatten the matrix
        array_matrix = array_matrix.flatten().tolist()
        self.assertEqual(array_matrix, [2, 0, 0, 0, 0, 1, 1, 0, 2])
    def test_confusion_real(self):
        true = [26,5,8,17,27,28]
        pred = [26,5,8,17,27,27]
        confusion_matrix = np.zeros((29,29),dtype=int)
        array_matrix = confusion_matrix(confusion_matrix, true, pred)
        self.assertEqual(array_matrix,[])
if __name__ == '__main__':
    unittest.main()