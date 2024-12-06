import unittest
from ml_utils.data.nlp import *


class TestNLP(unittest.TestCase):
    def test_boolean_triangular_mask(self):
        flat_mask = torch.tensor([1,1,1,0,0], dtype=torch.bool)
        expected = torch.tensor([[[1,0,0,0,0],
                                  [1,1,0,0,0],
                                  [1,1,1,0,0],
                                  [0,0,0,0,0],
                                  [0,0,0,0,0]]], dtype=torch.bool)
        self.assertTrue(torch.allclose(boolean_triangular_mask(flat_mask), expected))


if __name__ == '__main__':
    unittest.main(verbosity=2)