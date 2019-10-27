import sys
import argparse
import configparser
from datetime import datetime
from loguru import logger

# Torch stuff
import torch
from torch.utils.data import DataLoader

# Project modules
from utilities.config import AttnGANConfig
from training.trainer import Trainer

"""
TODO -->
1. Add seed randomization for reproducible results
2. When we have a working generator, visualizations for the presentation:
    1. Visualize the attention with the attn_maps of the Attention model or the words_loss?
    2. Visualize images by the generator
3. Use hyperparameters from config to initialize models all over the code..

"""


@logger.catch
def main():
    # Some boilerplate code for easier logging.
    logger.remove(0)
    logger.add(sys.stdout, level="INFO")
    logger.add("../logs/{time}.log", level="DEBUG")

    start_time = datetime.now()
    logger.info("Started at: {}".format(start_time))

    # Parsing main arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_path', required=True, help='Path to the configuration.ini file to work with.')
    parser.add_argument('--train', action='store_true', default=False, help='Training the model')
    # parser.add_argument('--validation', action='store_true', default=False, help='Training the model')
    # parser.add_argument('--test', action='store_true', default=False, help='Testing the model')
    args = parser.parse_args()

    # Read Configuration file
    logger.info("Loading config file from path: {}".format(args.cfg_path))
    config = AttnGANConfig(args.cfg_path)
    config.log_config_info()

    # Pytorch stuff:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        dataset_name = config.dataset['name']
        logger.info("Starting training on {} dataset".format(dataset_name))
        # Training
        trainer = Trainer(config)
        trainer.train()


if __name__ == "__main__":
    main()
