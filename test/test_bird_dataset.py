import unittest
import os
from src.dataset.birds_dataset import BirdsDataset
from src.training.trainer import Trainer
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
# Run all tests using this command:
# `python -m unittest -v`


class TestBirdDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBirdDataset, self).__init__(*args, **kwargs)
        self.dataset_preprocess_path = os.path.join(os.getcwd(), "../test", "Example_Dataset")

    def test_len(self):
        # There are 5994 samples in train set:
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")
        self.assertEqual(len(bird_dataset), 6)

        # There are 5794 samples in test set:
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="test")
        self.assertEqual(len(bird_dataset), 3)

    def test_getitem(self):
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")
        self.assertEqual(bird_dataset[0].image_id, 18)
        self.assertEqual(bird_dataset[1].image_id, 1296)

        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="test")
        self.assertEqual(bird_dataset[0].image_id, 52)
        self.assertEqual(bird_dataset[1].image_id, 22)

    def test_dataset_train(self):
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")

        # Two images in the train set:
        class_id = 24
        class_name = "024.Red_faced_Cormorant"
        self.assertEqual(bird_dataset.img_id2class[1296], class_id)
        self.assertEqual(bird_dataset.class_id2class_name[class_id], class_name)

        class_id = 191
        class_name = "191.Red_headed_Woodpecker"
        self.assertEqual(bird_dataset.img_id2class[11235], class_id)
        self.assertEqual(bird_dataset.class_id2class_name[class_id], class_name)

        # image not in the train set:
        self.assertFalse(11208 in bird_dataset.img_id2class)

    def test_dataset_test(self):
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="test")

        # Two images in the test set:
        class_id = 1
        class_name = "001.Black_footed_Albatross"
        self.assertEqual(bird_dataset.img_id2class[22], class_id)
        self.assertEqual(bird_dataset.class_id2class_name[class_id], class_name)

        class_id = 191
        class_name = "191.Red_headed_Woodpecker"
        self.assertEqual(bird_dataset.img_id2class[11208], class_id)
        self.assertEqual(bird_dataset.class_id2class_name[class_id], class_name)

        # image not in the test set:
        self.assertFalse(11236 in bird_dataset.img_id2class)

    def test_dataset_loading(self):
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")
        dataloader = DataLoader(bird_dataset, batch_size=5)
        batch = next(iter(dataloader))
        self.assertEqual(batch.images[0].shape[0], 5)
        dataloader = DataLoader(bird_dataset, batch_size=3)
        batch = next(iter(dataloader))
        self.assertEqual(batch.images[0].shape[0], 3)

    def test_dataset_batch_types(self):
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")
        dataloader = DataLoader(bird_dataset, batch_size=4)
        batch = next(iter(dataloader))
        self.assertIsInstance(batch.images, list)
        self.assertIsInstance(batch.images[0], torch.Tensor)
        self.assertIsInstance(batch.caption[0], torch.Tensor)
        self.assertIsInstance(batch.caption_length[0], torch.Tensor)
        self.assertIsInstance(batch.image_id[0], torch.Tensor)
        self.assertIsInstance(batch.class_id[0], torch.Tensor)
    
    def test_batch_shapes(self):
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")
        dataloader = DataLoader(bird_dataset, batch_size=4)
        batch = next(iter(dataloader))
        self.assertEqual(len(batch.images), 1)
        self.assertEqual(batch.images[0].shape[0], 4)
        self.assertEqual(batch.images[0].shape[1], 3)
        self.assertEqual(batch.images[0].shape[2], 299)
        self.assertEqual(batch.images[0].shape[3], 299)
        self.assertEqual(batch.caption.shape[0], 4)
        self.assertEqual(batch.caption_length.shape[0], 4)
        self.assertEqual(batch.class_id.shape[0], 4)
        self.assertEqual(batch.image_id.shape[0], 4)

    def test_image_range(self):
        """ After the changes, images should be in the range of [-1, 1].
        This due to the build in normalization step"""
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train", 
                                    number_images=3, base_image_size=64)
        dataloader = DataLoader(bird_dataset, batch_size=4)
        batch = next(iter(dataloader))
        self.assertLessEqual(batch.images[0].max(), 1)
        self.assertGreaterEqual(batch.images[0].min(), -1)
        self.assertLessEqual(batch.images[1].max(), 1)
        self.assertGreaterEqual(batch.images[2].min(), -1)

    def test_num_images_per_sample(self):
        # 3 image per sample, batch_size != 1
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train",
                                    number_images=3, base_image_size=64)
        dataloader = DataLoader(bird_dataset, batch_size=4)
        batch = next(iter(dataloader))
        self.assertEqual(len(batch.images), 3)
        # 2 image per sample, batch_size != 1
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train",
                                    number_images=2, base_image_size=64)
        dataloader = DataLoader(bird_dataset, batch_size=2)
        batch = next(iter(dataloader))
        self.assertEqual(len(batch.images), 2)

        # 1 image per sample, batch_size != 1
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train",
                                    number_images=1, base_image_size=64)
        dataloader = DataLoader(bird_dataset, batch_size=2)
        batch = next(iter(dataloader))
        # 1 image per sample
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train",
                                    number_images=1, base_image_size=64)
        dataloader = DataLoader(bird_dataset, batch_size=1)
        batch = next(iter(dataloader))
        self.assertEqual(len(batch.images), 1)

    def test_images_sizes(self):
        # Images shapes without batch
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train",
                                    number_images=3, base_image_size=64)
        sample = bird_dataset[0]
        self.assertEqual(len(sample.images), 3)
        self.assertEqual(sample.images[0].shape[0], 3)
        self.assertEqual(sample.images[0].shape[1], 64)
        self.assertEqual(sample.images[0].shape[2], 64)
        self.assertEqual(sample.images[1].shape[0], 3)
        self.assertEqual(sample.images[1].shape[1], 128)
        self.assertEqual(sample.images[1].shape[2], 128)
        self.assertEqual(sample.images[2].shape[0], 3)
        self.assertEqual(sample.images[2].shape[1], 256)
        self.assertEqual(sample.images[2].shape[2], 256)
        # Images shapes with batch
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train",
                                    number_images=3, base_image_size=64)
        dataloader = DataLoader(bird_dataset, batch_size=4)
        batch = next(iter(dataloader))
        self.assertEqual(len(batch.images), 3)
        # image 1
        self.assertEqual(batch.images[0].shape[0], 4)  # Batch axis
        self.assertEqual(batch.images[0].shape[1], 3)
        self.assertEqual(batch.images[0].shape[2], 64)
        self.assertEqual(batch.images[0].shape[3], 64)
        # image 2
        self.assertEqual(batch.images[1].shape[0], 4)
        self.assertEqual(batch.images[1].shape[1], 3)
        self.assertEqual(batch.images[1].shape[2], 128)
        self.assertEqual(batch.images[1].shape[3], 128)
        # image 3
        self.assertEqual(batch.images[2].shape[0], 4)
        self.assertEqual(batch.images[2].shape[1], 3)
        self.assertEqual(batch.images[2].shape[2], 256)
        self.assertEqual(batch.images[2].shape[3], 256)
    
    def test_image_transform(self):
        # Tests BOTH data loader as batch and image transform options
        # If None:
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")
        dataloader = DataLoader(bird_dataset, batch_size=4)
        batch = next(iter(dataloader))
        self.assertEqual(batch.images[0].shape[0], 4)
        self.assertEqual(batch.images[0].shape[1], 3)
        self.assertEqual(batch.images[0].shape[2], 299)
        self.assertEqual(batch.images[0].shape[3], 299)

        # if transforms not None
        transform = Trainer.compose_image_transforms(299)
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train", image_transform=transform)
        dataloader = DataLoader(bird_dataset, batch_size=2)
        batch = next(iter(dataloader))
        self.assertEqual(batch.images[0].shape[0], 2)
        self.assertEqual(batch.images[0].shape[1], 3)
        self.assertEqual(batch.images[0].shape[2], 299)
        self.assertEqual(batch.images[0].shape[3], 299)
        self.assertLessEqual(batch.images[0].max(), 1)
        self.assertGreaterEqual(batch.images[0].min(), -1)

    def test_text_transform(self):
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")
        # make sure that the numeralized sentence is padded correctly 
        self.assertEqual(bird_dataset[0].caption[-1], bird_dataset.token2idx("<eos>"))
    
    def test_padding(self):
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")
        self.assertEqual(len(bird_dataset[0].caption), bird_dataset.max_caption_length + 1)

        captions = [ \
        "This is a long sentence with more than 20 words that require alot of tokens in order to explain how the bird looks and where it sits", \
        "This is a different long sentence with more than 20 words that require alot of tokens in order to explain how the bird looks and where it sits"]
        caption_indices, caption_len = bird_dataset.choose_and_process_caption(captions)
        self.assertEqual(caption_len, bird_dataset.max_caption_length)
        self.assertEqual(len(caption_indices), bird_dataset.max_caption_length+1) #<eos> token
    
    # def test_sos_eos_pad(self):
    #     bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")
    #     tokens = ["tokeb", "to", "append", "sos_eos..."]
    #     padded_tokens = bird_dataset._pad_sos_eos(tokens)
    #     self.assertEqual(padded_tokens[-1], "<eos>")

    def test_tokenizer(self):
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")
        sentence = "text to tokenize"
        tokens = bird_dataset._tokenize(sentence)
        self.assertListEqual(tokens, ["text", "to", "tokenize"])

    def test_tokens_indices_mapping(self):
        bird_dataset = BirdsDataset(self.dataset_preprocess_path, split="train")
        # Test unknown token is 0:
        unknown_index = bird_dataset.token2idx("<unk>")
        self.assertEqual(unknown_index, 0)

        # Test idx2token mapping
        self.assertEqual(bird_dataset.idx2token(0), "<unk>")
        self.assertEqual(bird_dataset.idx2token(1), "<eos>")
        invalid_idx = bird_dataset.num_words + 1
        self.assertEqual(bird_dataset.idx2token(invalid_idx), "<unk>")
        
        # Test token2idx mapping mapping
        self.assertEqual(bird_dataset.token2idx("this_is_not_known_token"), unknown_index)
        self.assertEqual(bird_dataset.token2idx("<eos>"), 1)

if __name__ == "__main__":
    unittest.main()
    