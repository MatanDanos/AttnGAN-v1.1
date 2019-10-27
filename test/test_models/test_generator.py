import os
import unittest

import torch

from src.models.generator import GeneratorNetwork

class TestGenerator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGenerator, self).__init__(*args, **kwargs)
        self.Generator = GeneratorNetwork(input_vector_size=200, generator_features_number=128, word_features_number=256, residual_stages_number=2, residual_layers_in_stage_number=2)
    
    
    def test_single_generator_forward_pass(self):
        batch_size = 4
        generator_input = torch.rand(batch_size, 200)
        word_embeddings_encoded = torch.rand(batch_size,256,20)
        # sentence_embeddings_encoded = torch.rand(batch_size, 256)
        # real_labels = torch.rand(batch_size, 1, 1, 1)
        # fake_labels = torch.rand(batch_size, 1, 1, 1)
        words_mask = torch.zeros((batch_size, word_embeddings_encoded.shape[2]), dtype=torch.uint8)
        images, att_maps = self.Generator(generator_input, word_embeddings_encoded, words_mask)

        # 3 stages:
        self.assertEqual(len(images), 3)
        self.assertEqual(len(att_maps), 2)

        # batch size:
        self.assertEqual(images[0].size(0), batch_size)
        self.assertEqual(att_maps[0].size(0), batch_size)
        # 3 color channels
        self.assertEqual(images[0].shape[1], 3)
        self.assertEqual(images[1].shape[1], 3)
        self.assertEqual(images[2].shape[1], 3)
        
        # img sizes
        self.assertEqual(images[0].shape[2], 64)
        self.assertEqual(images[0].shape[3], 64)
        self.assertEqual(images[1].shape[2], 128)
        self.assertEqual(images[1].shape[3], 128)
        self.assertEqual(images[2].shape[2], 256)
        self.assertEqual(images[2].shape[3], 256)



if __name__ == "__main__":
    unittest.main()
