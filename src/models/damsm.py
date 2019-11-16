import torch
import torch.nn as nn

from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder

from loguru import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Damsm(nn.Module):

    def __init__(self, vocab_size, hyperparameters):
        """Performs initialization of the DAMSM, which is comprised of the text and image encoders
        Parameters:
            vocab_size (int): number of words in the vocab object
            hyperparameters (dictionary):
                             dict that holds all damsm hyperparameters
                             that were loaded in the config file
            weights_path_to_load (str, default=None):
                            path to load pre-trained weights from

        TODO -->
        1. Use the hyperparameters parameter to corretly give the hyperparameters in initialize
        """
        super(Damsm, self).__init__()

        logger.info("Initializing the DAMSM model")
        self.hyperparameters = hyperparameters
        self.weights_path_to_load = self.hyperparameters['weights_path_to_load']

        # Initialize the encoders and load weights if valid path
        self.text_encoder = TextEncoder(vocab_size, init_value=self.hyperparameters['init_value'])
        self.image_encoder = ImageEncoder()
        if self.weights_path_to_load is not None:
            logger.info("Loading encoders weights from {}".format(self.weights_path_to_load))
            encoders_weights = torch.load(self.weights_path_to_load)
            self.text_encoder.load_state_dict(encoders_weights['text_encoder'])
            self.image_encoder.load_state_dict(encoders_weights['image_encoder'])

    def forward(self, captions=None, captions_lengths=None, images=None):
        """Forward along the text enoder or the image encoder
        
        Note:
            Implemented for completness, but used the object directly.
        """
        if captions and caption_length:
            return self.text_encoder(captions, captions_lengths)
        if images:
            return self.image_encoder(images)

