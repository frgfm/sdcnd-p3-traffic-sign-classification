import unittest
import numpy as np
from src.utils import convert_to_grayscale, normalize


class Tester(unittest.TestCase):

    def test_convert_to_grayscale(self):

        # RGB image
        input_t = (255. * np.random.rand(1, 32, 32, 3)).astype(np.uint8)

        out = convert_to_grayscale(input_t)
        # Check grayscale conversion
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, input_t.shape[:-1] + (1,))
        self.assertTrue(np.all(out <= 1.))

    def test_normalize(self):

        # RGB image
        input_t = np.random.rand(1, 32, 32, 1).astype(np.float32)

        out = normalize(input_t, 0.5, 2.)
        self.assertEqual(out.dtype, input_t.dtype)
        self.assertEqual(out.shape, input_t.shape)


if __name__ == '__main__':
    unittest.main()
