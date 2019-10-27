import os
import unittest

# import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.text_encoder import TextEncoder
from src.dataset.birds_dataset import BirdsDataset


class TestTextEncoder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTextEncoder, self).__init__(*args, **kwargs)

        # Paths
        self.dataset_preprocess_path = os.path.join(os.getcwd(), "../test", "Example_Dataset")

        # Initialize datasets
        self.bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")

        # Initialize TextEncoder
        self.text_encoder = TextEncoder(self.bird_dataset.vocabulary.freqs.B())

    def test_forward_single_shape(self):
        # Tests the output shapes for a batch size of 1 (single sample)
        batch_size = 1
        train_loader = iter(DataLoader(self.bird_dataset, batch_size=batch_size))
        item = next(train_loader)

        caption = item.caption
        length = item.caption_length
        max_length = max(length)
        output, (h_n, c_n) = self.text_encoder(caption, length)

        # Output shape
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], self.text_encoder.num_directions * self.text_encoder.hidden_dim)
        self.assertEqual(output.shape[2], max_length)

        # h_n shape
        self.assertEqual(h_n.shape[0], batch_size)
        self.assertEqual(h_n.shape[1], self.text_encoder.semantic_space_dim)

        # c_n shape
        self.assertEqual(c_n.shape[0], self.text_encoder.num_layers * self.text_encoder.num_directions)
        self.assertEqual(c_n.shape[1], batch_size)
        self.assertEqual(c_n.shape[2], self.text_encoder.hidden_dim)

    def test_forward_batch_shape(self):
        # Tests the output shapes for a batch size of 4 (4 samples)
        batch_size = 4
        train_loader = iter(DataLoader(self.bird_dataset, batch_size=batch_size))
        item = next(train_loader)
        captions = item.caption
        lengths = item.caption_length
        max_length = max(lengths)
        output, (h_n, c_n) = self.text_encoder(captions, lengths)

        # Output shape
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], self.text_encoder.num_directions * self.text_encoder.hidden_dim)
        self.assertEqual(output.shape[2], max_length)

        # h_n shape
        self.assertEqual(h_n.shape[0], batch_size)
        self.assertEqual(h_n.shape[1], self.text_encoder.semantic_space_dim)

        # c_n shape
        self.assertEqual(c_n.shape[0], self.text_encoder.num_layers * self.text_encoder.num_directions)
        self.assertEqual(c_n.shape[1], batch_size)
        self.assertEqual(c_n.shape[2], self.text_encoder.hidden_dim)

    def test_hidden_dim(self):
        # Change the hidden dimension and the bidirectional option and make sure the dimension stay the same
        text_encoder = TextEncoder(self.bird_dataset.vocabulary.freqs.B(), 
                                   hidden_dim=512, bidirectional=True)
        self.assertEqual(text_encoder.hidden_dim, 512//2)
        self.assertEqual(type(text_encoder.hidden_dim), int)

        text_encoder = TextEncoder(self.bird_dataset.vocabulary.freqs.B(),
                                   hidden_dim=512, bidirectional=False)
        self.assertEqual(text_encoder.hidden_dim, 512)
        self.assertEqual(type(text_encoder.hidden_dim), int)

if __name__ == "__main__":
    unittest.main()
