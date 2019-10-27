"""The DAMSM Trainer Object file """

# General imports
import os
import numpy as np
from datetime import datetime

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Logging
from loguru import logger

# Project modules
from dataset.birds_dataset import BirdsDataset
# from models.text_encoder import TextEncoder
# from models.image_encoder import ImageEncoder
from models.damsm import Damsm
from losses.DAMSM_loss import damsm_loss
from utilities.early_stopping import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DAMSM_Trainer():

    def __init__(self, dataloaders, hyperparameters):
        """This is the trainer for the DAMSM component
        Receives the damsm_hyperparameters and dataloader as parameters, 
        """
        self.train_dataloader = dataloaders['train']
        self.validation_dataloader = dataloaders['validation']
        self.vocab = self.train_dataloader.dataset.vocabulary

        self.hyperparameters = hyperparameters
        self.num_epochs = self.hyperparameters['epochs']
        self.batch_size = self.hyperparameters['batch_size']
        self.gammas = {
            '1': self.hyperparameters['gamma1'],
            '2': self.hyperparameters['gamma2'],
            '3': self.hyperparameters['gamma3']
        }
        self.min_grad_to_clip = self.hyperparameters['min_grad_to_clip']
        self.log_tensorboard = self.hyperparameters['log_tensorboard']
        self.weights_dirpath_to_save = os.path.dirname(self.hyperparameters['weights_dirpath_to_save'])
        self.weights_path_to_load = self.hyperparameters['weights_path_to_load']

        self.record_rate = self.hyperparameters['record_rate']
        self.start_epoch = 1
        # Tensorboard Writer will output to ./runs/ directory by default
        # Will be False if test runs
        tensorboard_dir = self.hyperparameters['log_tensorboard']
        self.tensorboard_writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None

        # Init encoders
        self.damsm = Damsm(self.vocab.freqs.B(), self.hyperparameters).to(device)
        # self.text_encoder, self.image_encoder = self.init_encoders(self.vocab.freqs.B(), 
        #                                                            self.hyperparameters)
        
        # Early stopping object:
        self.early_stopping_patience = self.hyperparameters['early_stopping_patience']
        if self.early_stopping_patience:
            self.early_stopping = EarlyStopping(patience=self.early_stopping_patience)

        logger.info("Created a DAMSM_Trainer Object.")

    def init_optimizers(self, optimizer='Adam'):
        """Initialize an optimizer.
            default: Adam optimizer
        """
        # params = nn.ParameterList()
        # params.extend(self.text_encoder.parameters())
        # params.extend(self.image_encoder.parameters())
        
        # Chooses optimizer
        if optimizer.lower() == 'adam':
            learning_rate = self.hyperparameters['encoder_learning_rate']
            betas = self.hyperparameters['adam_betas']
            optimizer = optim.Adam(self.damsm.parameters(), learning_rate, betas)
        else:
            raise NotImplementedError("The given {} optimizer is not implemented".format(optimizer))
        
        if self.weights_path_to_load is not None:
            encoders_weights = torch.load(self.weights_path_to_load)
            optimizer.load_state_dict(encoders_weights['optimizer'])
            self.start_epoch = encoders_weights['epoch']

        return optimizer

    def train(self):
        """This is the pre-training stage of the DAMSM that was mentioned in the paper 
        i.e. will train the image and text encoder before training the actual generator
        Returns the trained text and image encoders.

        This method will also save the parameters of the trained models for later use

        """
        logger.info("Started training DAMSM...")
        
        # Initilizations
        optimizer = self.init_optimizers()
        total_steps = len(self.train_dataloader)
        min_val_loss = np.inf
        minibatch_indices = torch.LongTensor(range(self.batch_size)).to(device)

        logger.info("Started DAMSM training loop at {}...".format(datetime.now()))
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # ================================================================== #
            #               1) Use all training samples (single epoch)           #
            # ================================================================== #
            total_word_loss = 0
            total_sentence_loss = 0
            total_damsm_loss = 0
            self.damsm.train()
            # self.text_encoder.train()
            # self.image_encoder.train()

            for i, batch in enumerate(self.train_dataloader):
                optimizer.zero_grad()

                # ================================================================== #
                #                        1.1) Prepare Data                           #
                # ================================================================== #
                images, captions, captions_length, labels, _  = BirdsDataset.prepare_batch_date(batch)

                # ================================================================== #
                #                        1.2) Encode Data                            #
                # ================================================================== #
                encoded_words, (encoded_sentence, _) = self.damsm.text_encoder(captions, captions_length)
                encoded_images_subregions, encoded_image = self.damsm.image_encoder(images[0])

                # ================================================================== #
                #                        1.3) Compute the DAMSM loss                 #
                # ================================================================== #
                batch_damsm_losses = damsm_loss(encoded_sentence, encoded_image,
                                                encoded_words, encoded_images_subregions,
                                                captions_length, minibatch_indices, self.gammas)
                batch_damsm_loss, batch_words_loss, batch_sentence_loss = batch_damsm_losses

                # # ================================================================== #
                # #                        1.4) Backward, step, zero_grad              #
                # # ================================================================== #
                batch_damsm_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.damsm.text_encoder.parameters(), self.min_grad_to_clip)
                optimizer.step()

                # ================================================================== #
                #            1.6) Logging (to console to show progress)              #
                # ================================================================== #
                if (i + 1) % self.record_rate == 0:
                    logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, self.num_epochs, i + 1, total_steps, batch_damsm_loss.item()))
                total_word_loss += batch_words_loss
                total_sentence_loss += batch_sentence_loss
                total_damsm_loss += batch_damsm_loss

            # ================================================================== #
            #                        2) Validate Models                          #
            # ================================================================== #
            val_word_loss, val_sentence_loss, val_damsm_loss = self.validate(epoch, minibatch_indices)

            # ================================================================== #
            #                        3) Logging (tensorboard + arrays)           #
            # ================================================================== #
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('train/word_loss', total_word_loss, global_step=epoch)
                self.tensorboard_writer.add_scalar('train/sentence_loss', total_sentence_loss, global_step=epoch)
                self.tensorboard_writer.add_scalar('train/damsm_loss', total_damsm_loss, global_step=epoch)
                self.tensorboard_writer.add_scalar('validation/word_loss', val_word_loss, global_step=epoch)
                self.tensorboard_writer.add_scalar('validation/sentence_loss', val_sentence_loss, global_step=epoch)
                self.tensorboard_writer.add_scalar('validation/damsm_loss', val_damsm_loss, global_step=epoch)
        
            # ================================================================== #
            #                        4) Checkpoint if improved                   #
            # ================================================================== #
            if val_damsm_loss < min_val_loss:
                min_val_loss = val_damsm_loss
                logger.info("Checkpointing with {:.4} damsm validation loss...".format(val_damsm_loss))
                self.save_parameters(optimizer, epoch, total_damsm_loss, val_damsm_loss,
                                     os.path.join(self.weights_dirpath_to_save, "checkpoint_encoders.pt"))

            # ================================================================== #
            #               5) Stop Training if conditions are met               #
            # ================================================================== #
            if self.early_stopping_patience and self.early_stopping.step(min_val_loss.item()):
                break

        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        # When all epochs are done, save results for reproducibility
        self.save_parameters(optimizer, epoch, total_damsm_loss, val_damsm_loss,
                             os.path.join(self.weights_dirpath_to_save, "last_epoch_encoders.pt"))

        logger.info("Done training DAMSM...")
        return self.damsm.text_encoder, self.damsm.image_encoder

    @torch.no_grad()
    def validate(self, epoch, minibatch_indices):
        self.damsm.eval()
        # self.image_encoder.eval()
        # self.text_encoder.eval()
        
        val_word_loss = 0
        val_sentence_loss = 0
        val_damsm_loss = 0
        for i, batch in enumerate(self.validation_dataloader):
            # ================================================================== #
            #                        1.1) Prepare Data                           #
            # ================================================================== #
            images, captions, captions_length, labels, _  = BirdsDataset.prepare_batch_date(batch)

            # ================================================================== #
            #                        1.2) Encode Data                            #
            # ================================================================== #
            encoded_words, (encoded_sentence, _) = self.damsm.text_encoder(captions, captions_length)
            encoded_images_subregions, encoded_image = self.damsm.image_encoder(images[0])

            # ================================================================== #
            #                        1.3) Perform word + sentence loss           #
            # ================================================================== #
            losses = damsm_loss(encoded_sentence, encoded_image,
                                encoded_words, encoded_images_subregions,
                                captions_length, minibatch_indices, self.gammas)
            batch_damsm_loss, batch_word_loss, batch_sent_loss = losses

            # ================================================================== #
            #                        1.4) Record Validaton Loss                  #
            # ================================================================== #
            val_word_loss += batch_word_loss
            val_sentence_loss += batch_sent_loss
            val_damsm_loss += batch_damsm_loss

            if i > 50:
                break

        logger.info('Validation: Epoch [{}/{}], DAMSM Loss: {:.4f}'.format(epoch, self.num_epochs, val_damsm_loss.item()))
        return val_word_loss, val_sentence_loss, val_damsm_loss

    def save_parameters(self, optimizer, epoch, train_loss, val_loss, path_to_params):
        """Saves the encoders parameters and all relevant values in the path specified
            Note: 
                The parameters are saved and not the objects themselves.
                Thus loading should be done after the model was initialized,
                and by calling the load_state_dict method. """
        torch.save({
                    "text_encoder": self.damsm.text_encoder.state_dict(),
                    "image_encoder": self.damsm.image_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                    }, path_to_params)
