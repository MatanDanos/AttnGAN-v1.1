import unittest

import torch

from src.models.conditional_augmentation import ConditioningAugmentation


class TestConditionalAugmentation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestConditionalAugmentation, self).__init__(*args, **kwargs)
        # Initialize TextEncoder
        self.conditional_augmentation = ConditioningAugmentation()

    def test_single_forward_pass(self):
        # Bath size of 1
        batch_size = 1
        h = torch.rand(batch_size, self.conditional_augmentation.input_dim)
        c, mu, var = self.conditional_augmentation(h)

        self.assertEqual(c.shape[0], batch_size)
        self.assertEqual(c.shape[1], self.conditional_augmentation.condition_dim)

    def test_batch_forward_pass(self):
        # Bath size of 4
        batch_size = 4
        h = torch.rand(batch_size, self.conditional_augmentation.input_dim)
        c, mu, var = self.conditional_augmentation(h)

        self.assertEqual(c.shape[0], batch_size)
        self.assertEqual(c.shape[1], self.conditional_augmentation.condition_dim)


if __name__ == "__main__":
    unittest.main()
