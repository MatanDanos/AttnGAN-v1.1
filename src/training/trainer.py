# General
from datetime import datetime
import time

# Pytorch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Logging
from loguru import logger

# The AttnGAN sub-models:
from dataset.birds_dataset import BirdsDataset
from models.damsm import Damsm
from training.DAMSM_trainer import DAMSM_Trainer
from training.AttnGAN_trainer import AttnGAN_Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
    """
    Implements the Trainer class.
    This class will control the entire training process of the AttnGAN
    
    This class will:
    1. Record the train and validation loss during training
    2. Save the weights of the trained models in the models directory

    Parameters:
        config (Config): loaded Config instance
        validate (boolean): boolean that indicates whether to perform validation or not
            

    Output:
        trained AttnGAN model
    """
    def __init__(self, config):
        """
        Parameters:
            config (AttnGANConfig) - the loaded config file
        """
        # Store the inputs
        self.config = config

        # General hyperparameters
        self.damsm_hyperparameters = self.config.damsm_hyperparameters
        self.attngan_hyperparameters = self.config.attngan_hyperparameters
        self.dataset_path = self.config.dataset['path']
        
        # Booleans that indicate if we should train the DAMSM and AttnGAN
        self.train_DAMSM = self.damsm_hyperparameters['train']
        self.train_AttnGAN = self.attngan_hyperparameters['train']

        # Paths
        self.encoders_weights_path = self.damsm_hyperparameters['weights_path_to_load']
        self.attngan_weights_path = self.attngan_hyperparameters['weights_path_to_load']
        logger.info("Created a Trainer Object.")

    ####################### General Train Methods ####################### 
    def train(self):
        """This method will train the entire model.
        i.e. will train both DAMSM and AttnGAN components"""
        logger.info("Started training at {}...".format(datetime.now()))
        start_time = time.time()

        # Pretrain the DAMSM component (only if needed)
        if self.train_DAMSM:
            logger.info("Started pretraining the DAMSM component...")
            damsm_dataloaders = self.get_dataloader("DAMSM")
            damsm_trainer = DAMSM_Trainer(damsm_dataloaders, self.damsm_hyperparameters)
            text_encoder, image_encoder = damsm_trainer.train()

        # Train the AttnGAN component
        if self.train_AttnGAN:
            attngan_dataloaders = self.get_dataloader("ATTNGAN")
            if not self.train_DAMSM:
                logger.info("DAMSM pretraining skipped. Loading pretrained DAMSM models...")
                damsm = Damsm(attngan_dataloaders['train'].dataset.vocabulary.freqs.B(), self.damsm_hyperparameters).to(device)
                text_encoder = damsm.text_encoder
                image_encoder = damsm.image_encoder
            logger.info("Started training the AttnGAN component...")
            attnGAN_trainer = AttnGAN_Trainer(attngan_dataloaders, self.attngan_hyperparameters, text_encoder, image_encoder)
            attngan = attnGAN_trainer.train()
        else:
            logger.info("AttnGAN training skipped...")

        end_time = time.time()
        logger.info("Finished training after {} seconds".format(int(end_time - start_time)))

    def get_dataloader(self, dataset_type):
        """Returns the dataloaders as a dictionary for each dataloader
        i.e. if the dataset is intended to be used by the DAMSM or AttnGAN components
        Also, support train or test splits.
        """
        max_caption_length = self.damsm_hyperparameters['max_caption_length']
        if dataset_type == 'DAMSM':
            batch_size = self.damsm_hyperparameters['batch_size']
            base_img_size = self.damsm_hyperparameters['base_image_size']
            image_transforms = self.compose_image_transforms(base_img_size)
            train_dataset = BirdsDataset(self.dataset_path, split='train', image_transform=image_transforms,
                                         base_image_size=base_img_size, number_images=1,
                                         text_transform=None, max_caption_length=max_caption_length)
            validation_dataset = BirdsDataset(self.dataset_path, split='test', image_transform=image_transforms,
                                              base_image_size=base_img_size, number_images=1,
                                              text_transform=None, max_caption_length=max_caption_length)
        elif dataset_type == 'ATTNGAN':
            batch_size = self.attngan_hyperparameters['batch_size']
            base_img_size = self.attngan_hyperparameters['base_image_size']
            num_images = self.attngan_hyperparameters['attngan_size']
            image_transforms = self.compose_image_transforms(base_img_size)
            train_dataset = BirdsDataset(self.dataset_path, split='train', image_transform=image_transforms,
                                   base_image_size=base_img_size, number_images=num_images,
                                   text_transform=None, max_caption_length=max_caption_length)
            validation_dataset = BirdsDataset(self.dataset_path, split='test', image_transform=image_transforms,
                                   base_image_size=base_img_size, number_images=num_images,
                                   text_transform=None, max_caption_length=max_caption_length)
        else:
            raise ValueError("Invalid datasest_type was given. Excpeting 'DAMSM' or 'ATTNGAN. Got: {}".format(dataset_type))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        dataloaders = {"train": train_dataloader,
                       "validation": validation_dataloader}
        return dataloaders

    @staticmethod
    def compose_image_transforms(base_image_size):
        """Returns composed image transforms for using on PIL images"""
        resize_factor_for_cropping = 76 / 64 # TODO Understand why they hardcodes this value
        new_size = tuple(2*[int(base_image_size * resize_factor_for_cropping)])
        image_transforms = transforms.Compose([transforms.Resize(new_size),
                                               transforms.RandomCrop(base_image_size),
                                               transforms.RandomHorizontalFlip()
                                               ])
        return image_transforms

    def load_encoders_parameters(self, num_words):
        """ Will load the encoders parameters and return the trained objects
        Parameters:
            num_words (int): number of words in the vocabulary of the dataset
        Return:
            text_encoder, image_encoder
        """
        # Initialize the encoders
        text_encoder, image_encoder = DAMSM_Trainer.init_encoders(num_words, self.damsm_hyperparameters)

        # Load and use the state_dict for the encoders
        encoders_parameters = torch.load(self.encoders_weights_path)
        text_encoder.load_state_dict(encoders_parameters['text_encoder'])
        image_encoder.load_state_dict(encoders_parameters['image_encoder'])
        return text_encoder, image_encoder
