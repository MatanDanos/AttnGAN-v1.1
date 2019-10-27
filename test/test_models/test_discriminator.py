import os
import unittest

import torch

from src.models.discriminator import Discriminator

class TestGenerator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGenerator, self).__init__(*args, **kwargs)
        self.sizes = [64, 128, 256, 512]
        self.discriminators = []
        for size in self.sizes:
            self.discriminators.append(Discriminator(input_size=size))

    def test_single_discriminator_forward_pass_shapes(self):
        for discriminator, size in zip(self.discriminators, self.sizes):
            images = torch.rand((20, 3, size, size))
            conditions = torch.rand((20, 256))
            predictions, _ = discriminator(images, conditions)
            self.assertEqual(predictions.shape, (20, 2))

    def test_downsampler_creation(self):
        block = Discriminator._get_downsampler(3, 64, 16)
        self.assertEqual(next(block.parameters()).shape, torch.Size([64, 3, 4, 4]))
        self.assertEqual(len(list(block.children())), 11)

        block = Discriminator._get_downsampler(16, 64, 2)
        self.assertEqual(next(block.parameters()).shape, torch.Size([64, 16, 4, 4]))
        self.assertEqual(len(list(block.children())), 3)

    def test_scale_invariant_conv_layer(self):
        block16 = Discriminator._get_scale_invariant_conv_layer(3, 16)
        images = torch.rand(4, 3, 64, 64)
        output = block16(images)
        self.assertEqual(output.shape, torch.Size([4,16,64,64]))
        
        block32 = Discriminator._get_scale_invariant_conv_layer(64, 32)
        images = torch.rand(8, 64, 128, 128)
        output = block32(images)
        self.assertEqual(output.shape, torch.Size([8,32,128,128]))
        
        block64 = Discriminator._get_scale_invariant_conv_layer(3, 64)
        images = torch.rand(1, 3, 256, 256)
        output = block64(images)
        self.assertEqual(output.shape, torch.Size([1,64,256,256]))



if __name__ == "__main__":
    unittest.main()
