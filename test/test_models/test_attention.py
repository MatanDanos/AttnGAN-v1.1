import os
import unittest

import numpy as np
import torch

from src.models.attention import Attention


class TestAttention(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAttention, self).__init__(*args, **kwargs)
        self.attention = Attention()

    def test_forward_single_shape(self):
        # Tests the output shapes for a batch size of 1 (single sample)
        self._test_shapes(1)

    def test_forward_batch_shape(self):
        # Tests the output shapes for a batch size of 4 (4 samples)
        self._test_shapes(4)

    def _test_shapes(self, batch_size, seq_len=20):
        """Test the shape of the attention outputs for 3 possible image sizes a a given batch size"""
        image_shapes = [(64, 64), (128, 128), (256, 256)]
        for shape in image_shapes:
            image_features = torch.rand(batch_size, 128, *shape)
            word_features = torch.rand(batch_size, 256, seq_len)
            words_mask = torch.zeros((batch_size, seq_len), dtype=torch.uint8)
            excpected_N = np.prod(shape)
            attn, attn_coeff = self.attention(word_features, image_features, words_mask)
            self.assertEqual(attn.shape, torch.Size([batch_size, 128, excpected_N]))
            self.assertEqual(attn_coeff.shape, torch.Size([batch_size, excpected_N, seq_len]))

    def test_differnt_seq_len(self):
        for i in range(3):
            # test 3 random sequence lengths
            seq_len = torch.randint(2, 20, [1]).item()
            self._test_shapes(1, seq_len)
            self._test_shapes(4, seq_len)



if __name__ == "__main__":
    unittest.main()
