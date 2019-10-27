import os
import numpy as np
import pandas as pd

# Torch stuff..
import torch
from torch.utils.data import Dataset

# Image processing
from PIL import Image
from torchvision import transforms

# Text processing
from torchtext.vocab import Vocab
from nltk import RegexpTokenizer
from nltk.probability import FreqDist

from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BirdsDataset(Dataset):
    """Birds Dataset Class
        Used to iterate on the birds (CUB_200_2011) dataset.
        TODO -->
        1. Change to <eos> token to zero if any convergence issues
    """

    def __init__(self, root_dir, split="train",
                 image_transform=None, base_image_size=299, number_images=1,
                 text_transform=None, max_caption_length=18):
        """
        Init function to initialize the dataset.
        Parameters:
            * root_dir (string): Directory with all the images.
                root_dir should have the following:
                2 csv files: `images.csv` and `classes.csv` files
                2 subdirectories: `images` and `text` that have all the data inside
            * split (string): `train` or `test` for train data or test data respectively.
            Image related parameters:
                * image_transform (Transform object): pytorch transform object to be applied on an image, 
                                                    might be composed from a set of image transforms.
                * base_image_size (int): the size of the first image in the images list
                * number_images (int): the number of images in the images list
                                     The size of each image in the list will be a multiple of 2
                                     of the base_image_size value
            Text related parameters:
                * text_transform (Transform object): pytorch transform object to be applied on an image,
                                                   might be composed from a set of text transforms.
                * max_caption_length (int): Maximum number of words in a sentece.
                                          Default to 18 to allow for 2 more <sos> and <eos> strings
        """
        # Paths
        self.root_dir = root_dir
        self.images_csv_path = os.path.join(root_dir, "images.csv")
        self.classes_csv_path = os.path.join(root_dir, "classes.csv")
        self.images_dir_path = os.path.join(root_dir, "images")
        self.text_dir_path = os.path.join(root_dir, "text")

        # Train or Test data
        self.split = split

        # Dataframes
        class_df = pd.read_csv(self.classes_csv_path)
        images_df = pd.read_csv(self.images_csv_path)
        self.images_df = images_df[images_df.is_train == (0 if self.split == "test" else 1)].drop("is_train", axis=1).drop("Unnamed: 0", axis=1)

        # Look up tables
        self.img_id2class = {tup.image_id: tup.class_id for tup in self.images_df.itertuples()}
        self.img_id2path = {tup.image_id: tup.path[:-4] for tup in self.images_df.itertuples()}
        self.class_id2class_name = {row.class_id: row.class_name for row in class_df.itertuples()}
        self.class_name2class_id = {row.class_name: row.class_id for row in class_df.itertuples()}

        # Image related properties
        self.image_transform = image_transform
        self.base_image_size = base_image_size
        self.num_images = number_images
        self.images_size = self.base_image_size * (2 ** np.arange(self.num_images)) # [64, 128, 256] ...
        self.normalize_transform = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                                            (0.5, 0.5, 0.5))])
        self.resize_transforms = [transforms.Resize((size.item(), size.item())) for size in self.images_size]

        # Text related properties
        self.text_transform = text_transform
        self.max_caption_length = max_caption_length
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.vocabulary, self.num_words, self.longest_sentence_length = self._create_vocabulary()
        self.vocab_size = self.vocabulary.freqs.B()

    def __len__(self):
        """Returns the number of the samples in the dataset """
        return len(self.images_df)

    def __getitem__(self, idx):
        """
        @Parameters:
            idx (integer): Index in the dataset, 0 <= idx < dataset_size
        @Returns:
        A dictionary which holds all relevant data for a single sample after applying all transforms
            images: a list of tensors that represent the images in different sizes
            caption: A single caption chosen at random from the list of all 10 possible captions.
                The caption is already numeralized, and padded with <sos> and <eos>
            caption_length: A single tensor holding the length of the caption 
                            Note: Length does not include the <sos> <eos> tokens!
                            (need to +2 to get the actual length)
            image_id: the ID number of the image
            class_id: the class ID of the image
        """
        # The images
        images = self.load_images(idx)
        
        # The caption
        caption_indices, caption_length = self.get_caption(idx)

        # Image ID and class
        image_id = torch.tensor(self.images_df.iloc[idx].image_id)
        class_id = torch.tensor(self.images_df.iloc[idx].class_id)

        # Named tuple for easier access of the samples
        named_tup = namedtuple('BirdDataItem', ['images', 'caption', 'caption_length', 'class_id', 'image_id'])
        return named_tup(images, caption_indices, caption_length, class_id, image_id)

    ################################ Text Methods ################################
    def idx2token(self, idx):
        """Given an index (0<=int<vocab_size), returns the token that is represented by the index"""
        if not (0 <= idx and idx < self.num_words):
            return "<unk>"
        return self.vocabulary.itos[idx]

    def token2idx(self, token):
        """Given an token (string), returns the index that is represented by the token
        No check necessary since itos is implemented as defaultDict and thus will not throw
        an error in case of an unknown token, and will just return 0 (the index of unknown tokens).
        TODO -->
        1. When new torchtext version is out, this should be fixed according to the given <unk> location.
        """
        return self.vocabulary.stoi[token]

    def choose_and_process_caption(self, sample_captions):
        """Given a list of all 10 sample captions as string sentences, returns a single caption
        that was chosen at random.
        Will then proceeed to process the caption as follows:
        1. Choose a random caption
        2. Process the caption if a text_transform was given
        3. Tokenize the caption into single tokens (words)
        4. If the caption length is longer than the maximum, choose max_len random tokens.
        5. Pads <sos> and <eos> at the beginning and end of sentence
        6. Numeralize all tokens
        7. Pads <eos> integer value until the maximum length is reached and return the tensor

        Parameters:
            sample_captions (list of strings): All 10 captions of a given sample
        Returns:
            numeralized_caption (torch.LongTensor shape:(20,) ): All tokens in numerical form
            caption_length (int): the length of the original caption (excluding <sos> and <eos>)
        """
        # 1.
        rand_caption_idx = torch.randint(0, len(sample_captions), (1,)).item()
        caption = sample_captions[rand_caption_idx]

        # 2.
        if self.text_transform:
            caption = self.text_transform(caption)

        # 3.
        tokenized_caption = np.array(self._tokenize(caption))
        caption_length = len(tokenized_caption)

        # 4. for efficiency, avoid using random and sorting when possible
        if caption_length > self.max_caption_length:
            chosen_tokens_indices = torch.sort(torch.randperm(caption_length)[:self.max_caption_length])[0]
            tokenized_caption = tokenized_caption[chosen_tokens_indices]
            caption_length = self.max_caption_length

        # 5.
        tokenized_caption = np.concatenate((tokenized_caption, ["<eos>"]), axis=0)
        # tokenized_caption = self._pad_sos_eos(tokenized_caption)

        # 6.
        numeralized_caption = torch.LongTensor([self.token2idx(token) for token in tokenized_caption])

        # 7.
        padding = self.token2idx("<eos>")*torch.ones(self.max_caption_length - caption_length, dtype=torch.int64)
        return torch.cat((numeralized_caption, padding), dim=0), torch.tensor(caption_length)

    def _tokenize(self, text):
        """Tokenizes English text from a string into a list of strings (tokens)"""
        return self.tokenizer.tokenize(text)

    def _create_vocabulary(self):
        """Analyze all the text sentences in the data set and create a vocabulary:
        1. The dataset vocabulary
        2. Number of words in the vocabulary
        3. Length of the longest sentence """
        frequencies = FreqDist()
        max_sentence_length = 0
        for idx in range(self.__len__()):
            txt_path = os.path.join(self.text_dir_path, self.images_df.iloc[idx].path + ".txt")
            with open(txt_path, "r") as f:
                for line in f:
                    tokens = [token.lower() for token in self.tokenizer.tokenize(line)]
                    if len(tokens) > max_sentence_length:
                        max_sentence_length = len(tokens)
                    frequencies.update(tokens)
        # Finally, create the vocabulary object from the torchtext library.
        vocabulary = Vocab(frequencies, min_freq=2, specials=["<unk>", "<eos>"])
        return vocabulary, len(vocabulary.itos), max_sentence_length

    def get_caption(self, idx):
        """Given an index of a image-text sample,
        randomly choose a image caption from the list of 10 possible captions """
        captions_path = os.path.join(self.text_dir_path, self.images_df.iloc[idx].path + ".txt")
        with open(captions_path, "r") as captions_file:
            image_captions = [line.strip("\n") for line in captions_file.readlines()]
        # Choose and process a single caption
        caption_indices, caption_length = self.choose_and_process_caption(image_captions)
        return caption_indices, caption_length

    ################################ Image Methods ################################
    def load_images(self, idx):
        """Given an sample index, will return a list of the sample image in all required sizes
        The image will also be transformed according to the supplied transforms object,
        and will be tranformed into tensor and normalized into [0,1] domain"""
        # Load original image..
        img_path = os.path.join(self.images_dir_path, self.images_df.iloc[idx].path + ".jpg")
        orig_image = Image.open(img_path).convert("RGB")

        # Transform images according to the given transforms (e.g. crop, flip etc..)
        if self.image_transform is not None:
            transformed_image = self.image_transform(orig_image)
        else:
            transformed_image = orig_image

        # Create the list of images in different sizes to return
        resized_images = [resize_transform(transformed_image) for resize_transform in self.resize_transforms]
        
        # Normalize images to [0,1]
        final_images = [self.normalize_transform(img) for img in resized_images]
        return final_images

    ################################ General Methods ################################
    @staticmethod
    def prepare_batch_date(batch):
        """Preparing the batch data for training"""
        images = [img.to(device) for img in batch.images]
        captions = batch.caption.to(device)
        captions_length = batch.caption_length.to(device)
        labels = batch.class_id.to(device)
        image_ids = batch.image_id.to(device)
        return images, captions, captions_length, labels, image_ids

    def draw_random_sample(self):
        """Randomly draw a single sample from the dataset"""
        return self.__getitem__(torch.randint(low=0, high=self.__len__(), size=[1]).item())