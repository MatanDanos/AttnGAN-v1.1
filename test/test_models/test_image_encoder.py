import os
import unittest

import torch

from src.models.image_encoder import ImageEncoder


class TestImageEncoder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestImageEncoder, self).__init__(*args, **kwargs)
        self.image_encoder = ImageEncoder()
        self.word_feature_dim = 256  # multi-modal semantic_space_dim
        self.num_local_features = 289  # Number of local image feature

    def test_reshaping(self):
        # Reshaping from smaller to larger image (normal case)
        batch_size = 1
        image = torch.rand(batch_size, 3, 256, 256)
        local_features, global_features = self.image_encoder(image)
        self._check_shape(local_features, global_features, batch_size)
        image = torch.rand(batch_size, 3, 128, 128)
        local_features, global_features = self.image_encoder(image)
        self._check_shape(local_features, global_features, batch_size)

        # Reshaping from larger to smaller image
        # (special case, if we use more than 2 generators..)
        image = torch.rand(batch_size, 3, 512, 512)
        local_features, global_features = self.image_encoder(image)
        self._check_shape(local_features, global_features, batch_size)
        image = torch.rand(batch_size, 3, 1024, 1024)
        local_features, global_features = self.image_encoder(image)
        self._check_shape(local_features, global_features, batch_size)

    def test_forward_single_shape(self):
        # Tests the output shapes for a batch size of 1 (single sample)
        batch_size = 1
        image_size = 299
        image = torch.rand(batch_size, 3, image_size, image_size)
        local_features, global_feature = self.image_encoder(image)
        self._check_shape(local_features, global_feature, batch_size)

    def test_forward_batch_shape(self):
        # Tests the output shapes for a batch size of 4 (4 samples)
        batch_size = 4
        image_size = 299
        image = torch.rand(batch_size, 3, image_size, image_size)
        local_features, global_feature = self.image_encoder(image)
        self._check_shape(local_features, global_feature, batch_size)

    def _check_shape(self, local_features, global_feature, batch_size):
        self.assertEqual(local_features.shape, torch.Size([batch_size, self.word_feature_dim, self.num_local_features]))
        self.assertEqual(global_feature.shape, torch.Size([batch_size, self.word_feature_dim]))


if __name__ == "__main__":
    unittest.main()
