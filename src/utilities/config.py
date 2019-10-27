import configparser
import os

from loguru import logger

class AttnGANConfig():
    """ This class holds all configurable variables
        The variables are read from config file given at startup
        All config variables needed to be set both here and in the config.ini file"""

    def __init__(self, config_filename):
        logger.info("Loading configurations from: {}".format(config_filename))
        self.config = configparser.ConfigParser()
        self.config.read(config_filename)

        self.general = {
            'name': self.config['general'].get("run_name"),
            'seed': self.config['general'].getint('seed'),
            'gpu': self.config['general'].getint('gpu')
        }

        self.dataset = {
            'name': self.config['datasets'].get('name'),
            'path':  self.config['datasets'].get('path')
        }

        self.damsm_hyperparameters = {
            'train': self.config['damsm_hyperparameters'].getboolean('train'),
            'min_word_frequency': self.config['damsm_hyperparameters'].getint('min_word_frequency'),
            'max_caption_length': self.config['damsm_hyperparameters'].getint('max_caption_length'),
            'embedding_dim': self.config['damsm_hyperparameters'].getint('embedding_dim'),
            'word_feature_dim': self.config['damsm_hyperparameters'].getint('word_feature_dim'),
            'general_sentence_dim': self.config['damsm_hyperparameters'].getint('general_sentence_dim'),
            'LSTM_hidden_dim': self.config['damsm_hyperparameters'].getint('LSTM_hidden_dim'),
            'base_image_size': self.config['damsm_hyperparameters'].getint('base_image_size'),
            'init_value': self.config['damsm_hyperparameters'].getfloat('LSTM_hidden_dim'),
            'dropout_proba': self.config['damsm_hyperparameters'].getfloat('dropout_proba'),
            'encoder_learning_rate': self.config['damsm_hyperparameters'].getfloat('encoder_learning_rate'),
            'adam_betas': eval(self.config['damsm_hyperparameters'].get('adam_betas')),
            'min_grad_to_clip': self.config['damsm_hyperparameters'].getfloat('min_grad_to_clip'),
            'gamma1': self.config['damsm_hyperparameters'].getfloat('gamma1'),
            'gamma2': self.config['damsm_hyperparameters'].getfloat('gamma2'),
            'gamma3': self.config['damsm_hyperparameters'].getfloat('gamma3'),
            'batch_size': self.config['damsm_hyperparameters'].getint('batch_size'),
            'epochs': self.config['damsm_hyperparameters'].getint('epochs'),
            'record_rate' : self.config['damsm_hyperparameters'].getint('record_rate'),
            'log_tensorboard': self.config['damsm_hyperparameters'].get('log_tensorboard'),
            'early_stopping_patience' : self.config['damsm_hyperparameters'].get('early_stopping_patience'),
            'weights_dirpath_to_save' : self.config['damsm_hyperparameters'].get('weights_dirpath_to_save'),
            'weights_path_to_load' : self.config['damsm_hyperparameters'].get('weights_path_to_load')
        }

        self.attngan_hyperparameters = {
            'train': self.config['attngan_hyperparameters'].getboolean('train'),
            'base_image_size': self.config['attngan_hyperparameters'].getint('base_image_size'),
            'condition_dim': self.config['attngan_hyperparameters'].getint('condition_dim'),
            'generator_l1_dim': self.config['attngan_hyperparameters'].getint('generator_l1_dim'),
            'discriminator_learning_rate': self.config['attngan_hyperparameters'].getfloat('discriminator_learning_rate'),
            'generator_learning_rate': self.config['attngan_hyperparameters'].getfloat('generator_learning_rate'),
            'gamma1': self.config['attngan_hyperparameters'].getfloat('gamma1'),
            'gamma2': self.config['attngan_hyperparameters'].getfloat('gamma2'),
            'gamma3': self.config['attngan_hyperparameters'].getfloat('gamma3'),
            'lambda': self.config['attngan_hyperparameters'].getfloat('lambda'),
            'batch_size': self.config['attngan_hyperparameters'].getint('batch_size'),
            'epochs': self.config['attngan_hyperparameters'].getint('epochs'),
            'attngan_size': self.config['attngan_hyperparameters'].getint('attngan_size'),  # number of Generators, discriminators in the architecture
            'adam_betas': eval(self.config['attngan_hyperparameters'].get('adam_betas')),
            'record_rate' : self.config['attngan_hyperparameters'].getint('record_rate'),
            'log_tensorboard': self.config['attngan_hyperparameters'].get('log_tensorboard'),
            # 'early_stopping_patience' : self.config['attngan_hyperparameters'].getint('early_stopping_patience'),
            'label_smooth': self.config['attngan_hyperparameters'].getfloat('label_smooth'),
            'd_noise_decay': self.config['attngan_hyperparameters'].getfloat('d_noise_decay'),
            'feature_matching': self.config['attngan_hyperparameters'].getboolean('feature_matching'),
            'weights_dirpath_to_save' : self.config['attngan_hyperparameters'].get('weights_dirpath_to_save'),
            'weights_path_to_load' : self.config['attngan_hyperparameters'].get('weights_path_to_load')
        }

    def log_config_info(self):
        logger.info("Logging all configurations for reproducibility:")
        for section in self.config.sections():
            logger.info("\nSection: {}\nValues: {}".format(section, self.config.items(section)))