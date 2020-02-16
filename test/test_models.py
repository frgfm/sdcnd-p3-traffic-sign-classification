import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from src.models import lenet5


class Tester(unittest.TestCase):

    def test_lenet5(self):

        # Create a model
        model = lenet5()
        self.assertIsInstance(model, Sequential)

        # Shape test
        input_t = np.random.rand(1, 32, 32, 1).astype(np.float32)
        out = model.predict(x=input_t)
        self.assertEqual(out.shape, (1, 43))
        self.assertTrue(np.all(out <= 1))


if __name__ == '__main__':
    unittest.main()
