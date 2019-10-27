import os
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Logging
from loguru import logger

# Our modules
from dataset.birds_dataset import BirdsDataset
from models.conditional_augmentation import ConditioningAugmentation
from models.generator import GeneratorNetwork
from models.discriminator import Discriminator
from losses.AttnGAN_loss import discriminator_loss, generator_loss, feature_matching
from losses.DAMSM_loss import damsm_loss


from visualizations.images import visualize_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttnGAN_Trainer():

    def __init__(self, dataloaders, hyperparameters, text_encoder, image_encoder):
        """This is the trainer for the AttnGAN component
            i.e. the red box in figure 2 in the paper including the conditional augmentation.
        Parameters:
            dataloaders (dictionary): dictionary that holds the train and validation dataloaders
            hyperparameters (dictionary): dictionary holding all AttnGAN hyperparameters
            text_encoder (TextEncoder): trained TextEncoder
            image_encoder (ImageEncoder): trained ImageEncoder

        Notation:
            All over the code:
                * The generators will be denoted with capital G,
                * The discriminators will be denoted with capital D,
                * The conditional augmentation will be denoted with CA or ca
        """
        # Store the inputs
        self.train_dataloader = dataloaders['train']
        self.validation_dataloader = dataloaders['validation']
        self.hyperparameters = hyperparameters
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        # Store derived values from inputs
        self.idx2str = self.train_dataloader.dataset.vocabulary.itos
        self.str2idx = self.train_dataloader.dataset.vocabulary.stoi

        # Set models hyper parameters
        self.condition_dim = self.hyperparameters['condition_dim']
        self.sementic_space_dim = self.text_encoder.semantic_space_dim

        # Set train hyper parameters
        self.attngan_size = self.hyperparameters['attngan_size']
        self.num_generators = self.attngan_size
        self.num_discriminators = self.attngan_size
        self.base_image_size = self.hyperparameters['base_image_size']
        self.images_sizes_list = self.base_image_size * (2 ** np.arange(self.attngan_size)) # [64, 128, 256] ...
        self.batch_size = self.hyperparameters['batch_size']
        self.num_epochs = self.hyperparameters['epochs']
        self.start_epoch = 1

        # from tips for training & Improved Techniques for Training GANs paper
        self.label_smoothing = self.hyperparameters['label_smooth']
        self.D_noise_decay = self.hyperparameters['d_noise_decay']
        self.feature_matching = self.hyperparameters['feature_matching']

        self.Lambda = self.hyperparameters['lambda']
        self.gammas = {
            '1': 5.,
            '2': 5.,
            '3': 10.
        }

        self.weights_dirpath_to_save = self.hyperparameters['weights_dirpath_to_save']
        self.weights_path_to_load = self.hyperparameters['weights_path_to_load']

        self.record_rate = self.hyperparameters['record_rate']
        # Tensorboard Writer will output to ./runs/ directory by default
        # Will be False if test runs
        tensorboard_dir = self.hyperparameters['log_tensorboard']
        self.tensorboard_writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None

        self.init_components()
        logger.info("Created a AttnGAN_Trainer Object.")

    def init_components(self):
        """
        Loads and initialize the model components.
        In that:
            1. moves the text and image encoder the evaluation mode by
                calling eval() and changing requires_grad to False
            2. Initialize the componenets: CA, G & D.
            3. All componenets will be moved to the active device (GPU or CPU)

        TODO -->
        1. Use the hyperparameters property to corretly give the hyperparameters in initialize
         """
        # Change the text encoder to prediction mode (eval() + requires_grad = False)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()
        self.text_encoder = self.text_encoder.to(device)

        # Change the text encoder to prediction mode (eval() + requires_grad = False)
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        self.image_encoder.eval()
        self.image_encoder = self.image_encoder.to(device)

        # Conditional Augmentation
        self.conditional_augmentation = ConditioningAugmentation()
        self.conditional_augmentation.train()
        self.conditional_augmentation = self.conditional_augmentation.to(device)

        # Discriminators 
        self.discriminators = []
        for i in range(self.num_discriminators):
            self.discriminators.append(Discriminator(self.images_sizes_list[i]))
            self.discriminators[i].train()
            self.discriminators[i] = self.discriminators[i].to(device)

        # Generator
        self.generator = GeneratorNetwork()
        self.generator.train()
        self.generator = self.generator.to(device)
        
        # Load weights if path was given
        if self.weights_path_to_load:
            logger.info("Loading models state dict from {}".format(self.weights_path_to_load))
            weights = torch.load(self.weights_path_to_load)
            self.text_encoder.load_state_dict(weights['models']['text_encoder'])
            self.image_encoder.load_state_dict(weights['models']['image_encoder'])
            self.conditional_augmentation.load_state_dict(weights['models']['conditional_augmentation'])
            self.generator.load_state_dict(weights['models']['generator'])
            for i, D in enumerate(self.discriminators):
                D.load_state_dict(weights['models']['discriminators'][i])
            self.start_epoch = weights['epoch']

    def init_optimizers(self):
        """Init the optimizers for the conditional augmentation, generator and discriminator
        1. Try different optimizers if time permits.
        """
        # Discriminator optimizers:
        D_optimizers = []
        for i in range(self.num_discriminators):
            D_optimizers.append(torch.optim.Adam(self.discriminators[i].parameters(),
                                                 lr=self.hyperparameters['discriminator_learning_rate'],
                                                 betas=self.hyperparameters['adam_betas']))

        # Generator optimizer (added the Conditional Augmentation to the same optimizer, because it's involved in the optimization process.)
        G_params = nn.ParameterList()
        G_params.extend(self.conditional_augmentation.parameters())
        G_params.extend(self.generator.parameters())
        G_optimizer = torch.optim.Adam(G_params, lr=self.hyperparameters['generator_learning_rate'],
                                       betas=self.hyperparameters['adam_betas'])

        if self.weights_path_to_load:
            logger.info("Loading optimizers state dict from {}".format(self.weights_path_to_load))
            weights = torch.load(self.weights_path_to_load)
            G_optimizer.load_state_dict(weights['optimizers']['G_optimizer'])
            for i, D in enumerate(D_optimizers):
                D.load_state_dict(weights['optimizers']['D_optimizers'][i])
            self.start_epoch = weights['epoch']
        return D_optimizers, G_optimizer

    def train(self):
        """Trains the AttnGAN model
            This method assumes that both text and image encoders are already pretrained, 
            and it will not train them.

        Saves all model components
        Returns the final trained generator.
        """
        logger.info("Started training the AttnGAN")

        # Optimizers:
        D_optimizers, G_optimizer = self.init_optimizers()
        # Labels for the loss functions
        base_real_labels, base_fake_labels = self.prepare_real_fake_labels(self.batch_size) # Make sure that drop_last=True in the dataloader, thus in all steps the same vector sizes
        minibatch_indices = torch.LongTensor(range(self.batch_size)).to(device)
        feature_matching_weight_logit = torch.tensor(-10.)
        # Logging
        total_steps = len(self.train_dataloader)
        # Checkpointing
        min_val_loss = np.inf
        # Constant attngan inputs for consistent visual evaluation 
            # fixed noise vector that we will use to create the same image every epoch for logging
        fixed_noise = torch.randn(1, self.condition_dim).to(device)
            # random train and validation sample for constant evaluation
        train_sample = self.train_dataloader.dataset.draw_random_sample()
        val_sample = self.validation_dataloader.dataset.draw_random_sample()

        logger.info("Started AttnGAN training loop at {} for {} epochs...".format(datetime.now(), self.num_epochs))
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # ================================================================== #
            #                    1)  Epoch Initializations                       #
            # ================================================================== #
            epoch_metrics = defaultdict(int)
            self.change_models_mode(mode='train')

            # Update feature matching loss weight 
            # Weight is increasing, to regularize when D & G are strong enough
            # After 100 epochs, it will be ~0.5, after 150 and forward it will be 1
            feature_matching_weight_logit += 0.1 
            feature_matching_weight = torch.sigmoid(feature_matching_weight_logit)

            # ================================================================== #
            #               2) Use all training samples (single epoch)           #
            # ================================================================== #
            for step, batch in enumerate(self.train_dataloader, 1):
                # ================================================================== #
                #                      2.1) Prepare Batch Data                       #
                # ================================================================== #
                images, captions, captions_length, labels, image_ids = BirdsDataset.prepare_batch_date(batch)

                # ================================================================== #
                #                      2.2) Encode Captions                          #
                # ================================================================== #
                encoded_words, (encoded_sentence, _) = self.text_encoder(captions, captions_length)
                encoded_words, encoded_sentence = encoded_words.detach(), encoded_sentence.detach()

                words_mask = captions == self.str2idx['<eos>']
                words_mask = words_mask[:, :encoded_words.size(2)]

                ## Conditional vector:
                conditioned_vec, mu, logvar = self.conditional_augmentation(encoded_sentence)
                z = torch.randn(conditioned_vec.shape[0], self.condition_dim).to(device)  # Noise vector
                gan_input = torch.cat([conditioned_vec, z], dim=1)  # the concatentated conditioning with the noise

                # ================================================================== #
                #                      2.3) Generate Fake Images                     #
                # ================================================================== #
                generated_images, att_maps = self.generator(gan_input, encoded_words, words_mask)

                # ================================================================== #
                #            2.4) # Tricks to improve training                       #
                #             (label smoothing (6) & noisy images (13))              #
                # ================================================================== #
                if self.label_smoothing != 0:
                    real_labels, fake_labels = self.prepare_label_smoothing(base_real_labels, base_fake_labels, smooth=self.label_smoothing)
                else:
                    real_labels, fake_labels = real_labels, fake_labels
                if self.D_noise_decay != 0:
                    generated_images = self.add_noise_to_images(generated_images)
                    images = self.add_noise_to_images(images)

                # ================================================================== #
                #               2.5) Train The Discriminators one by one             #
                # ================================================================== #
                self.change_requires_grad(self.discriminators, require_grad=True)
                D_real_feature_maps = []
                for i, (D_net, D_optimizer) in enumerate(zip(self.discriminators, D_optimizers)):
                    D_optimizer.zero_grad()

                    # Get prediction_probabilites of real and fake images and compute loss
                    real_probabilities, real_feature_maps = D_net(images[i], encoded_sentence)
                    D_real_feature_maps.append(real_feature_maps)
                    fake_probabilities, _ = D_net(generated_images[i].detach(), encoded_sentence)  # Detach to avoid the backward into the generator

                    # Training the discriminators on real images with mismatching sentnces
                    # To help D discriminate between right image-text pair
                    wrong_probabilities, _ = D_net(images[i][:-1], encoded_sentence[1:])  # Creating a mistmatch by using the first n-1 and last n-1 of the real features and their conditions
                    Di_loss = discriminator_loss(real_probabilities, fake_probabilities, wrong_probabilities, real_labels, fake_labels)
                    epoch_metrics['D_loss'] += Di_loss
                    # backprop
                    Di_loss.backward()
                    D_optimizer.step()

                # ================================================================== #
                #                    2.6) Train The Generator                        #
                # ================================================================== #
                self.change_requires_grad(self.discriminators, require_grad=False)
                D_proba_list = []
                D_fake_feature_maps = []
                for i, discriminator in enumerate(self.discriminators):
                    proba, feature_maps = discriminator(generated_images[i], encoded_sentence)
                    D_proba_list.append(proba)
                    D_fake_feature_maps.append(feature_maps)

                G_optimizer.zero_grad()
                step_G_loss = generator_loss(D_proba_list, real_labels)
                epoch_metrics['G_loss'] += step_G_loss

                if self.feature_matching:
                    # Adds the feature matching regularization
                    for real_map, fake_map in zip(D_real_feature_maps, D_fake_feature_maps):
                        feature_matching_loss = feature_matching(real_map, fake_map)
                        # step_G_loss += feature_matching_weight * feature_matching_loss
                        epoch_metrics['feature_matching_loss'] += feature_matching_loss

                # Encode the resulting last generated image
                encoded_images_subregions, encoded_image = self.image_encoder(generated_images[-1])
                
                # DAMSM loss for words and generated image
                all_damsm_losses = damsm_loss(encoded_sentence, encoded_image,
                                              encoded_words, encoded_images_subregions,
                                              captions_length, minibatch_indices, self.gammas)
                step_damsm_loss, step_word_loss, step_sent_loss = all_damsm_losses
                kl_loss = KL_loss(mu, logvar)  # KL loss found in their code, but was not mentioned in their paper

                # Final step loss
                step_loss = step_G_loss + (self.Lambda * step_damsm_loss) + kl_loss
                step_loss.backward()
                G_optimizer.step()
                # Log the lossed for tensorboard
                epoch_metrics['damsm_loss'] += step_damsm_loss
                epoch_metrics['word_loss'] += step_word_loss
                epoch_metrics['sentence_loss'] += step_sent_loss
                epoch_metrics['total_loss'] += step_loss

                # ================================================================== #
                #            2.7) Logging (to console to show progress)              #
                # ================================================================== #
                # Current step losses
                if step % self.record_rate == 0:
                    logger.info("""Epoch [{}/{}], Step [{}/{}] Losses:
                                    Discriminator: {:.4f},. Generator: {:.4f}, Damsm: {:.4f}.
                                """.format(epoch, self.num_epochs, step, total_steps,
                                epoch_metrics['D_loss'].item(), epoch_metrics['G_loss'].item(), epoch_metrics['damsm_loss'].item()))

                # temp images in steps
                if step % 300 == 0:
                    self.generate_new_image(epoch, step=step)

            # ================================================================== #
            #                        3) Validate results                         #
            # ================================================================== #
            validation_metrics = self.validate(epoch, real_labels, fake_labels, minibatch_indices)

            # ================================================================== #
            #                        4) Checkpoint if improved                   #
            # ================================================================== #
            if validation_metrics['total_loss'] < min_val_loss:
                min_val_loss = validation_metrics['total_loss']
                logger.info("Checkpointing the AttnGAN models with validation loss: {:.4f}".format(min_val_loss))
                self.save_attngan(epoch, G_optimizer, D_optimizers,
                                  epoch_metrics['total_loss'], validation_metrics['total_loss'],
                                  os.path.join(self.weights_dirpath_to_save, 'checkpoint_attngan.pt'))
            if epoch % 10 == 0:
                # Just a meantime save incase good images but bad loss
                logger.info("Checkpointing every 10 epochs...")
                self.save_attngan(epoch, G_optimizer, D_optimizers,
                                  epoch_metrics['total_loss'], validation_metrics['total_loss'],
                                  os.path.join(self.weights_dirpath_to_save, 'epoch_checkpoint_attngan.pt'))

            # ================================================================== #
            #                        5) Logging to tensorboard                   #
            # ================================================================== #
            if self.tensorboard_writer is not None:
                # Log all possible epoch losses
                self.tensorboard_log_scalers(epoch, 'train', epoch_metrics)
                self.tensorboard_log_scalers(epoch, 'validation', validation_metrics)
                # Generate images and log images:
                self.tensorboard_log_generated_images(epoch, fixed_noise, train_sample, val_sample)
                self.generate_new_image(epoch)
                

        # Final operations to finish training...
        self.save_attngan(epoch, G_optimizer, D_optimizers,
                          epoch_metrics['total_loss'], validation_metrics['total_loss'],
                          os.path.join(self.weights_dirpath_to_save, 'final_attngan.pt'))
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        logger.info("Done training the AttnGAN")
        
        # Returns the final trained generator
        return self.generator

    @torch.no_grad()
    def validate(self, epoch, real_labels, fake_labels, minibatch_indices):
        """Validation method."""
        self.change_models_mode(mode='eval')
        
        validation_metrics = defaultdict(int)
        for i, batch in enumerate(self.validation_dataloader):
            if i > 50:
                break # Train only on 50 batches of the validation dataloader
            # ================================================================== #
            #                      1) Prepare Batch Data                       #
            # ================================================================== #
            images, captions, captions_length, labels, image_ids = BirdsDataset.prepare_batch_date(batch)

            # ================================================================== #
            #                      2) Encode Captions                          #
            # ================================================================== #
            encoded_words, (encoded_sentence, _) = self.text_encoder(captions, captions_length)

            ## Conditional vector:
            conditioned_vec, mu, var = self.conditional_augmentation(encoded_sentence)
            z = torch.randn(conditioned_vec.shape[0], self.condition_dim).to(device)  # Noise vector
            gan_input = torch.cat([conditioned_vec, z], dim=1)  # the concatentated conditioning with the noise

            # ================================================================== #
            #                      2.3) Generate Fake Images                     #
            # ================================================================== #
            words_mask = captions == self.str2idx['<eos>']
            words_mask = words_mask[:, :encoded_words.size(2)]
            generated_images, att_maps = self.generator(gan_input, encoded_words, words_mask)

            # ================================================================== #
            #               2.4) Train The Discriminators one by one             #
            # ================================================================== #
            for i, D_net in enumerate(self.discriminators):
                # Get prediction_probabilites and compute loss
                real_probabilities, _ = D_net(images[i], encoded_sentence)
                fake_probabilities, _ = D_net(generated_images[i].detach(), encoded_sentence)  # Detach to avoid the backward into the generator
                wrong_probabilities, _ = D_net(images[i][:-1], encoded_sentence[1:])
                Di_loss = discriminator_loss(real_probabilities, fake_probabilities, wrong_probabilities, real_labels, fake_labels)
                validation_metrics['D_loss'] += Di_loss

            # ================================================================== #
            #                    2.5) Train The Generator                        #
            # ================================================================== #
            D_proba_list = []
            for i, discriminator in enumerate(self.discriminators):
                probas, _ = discriminator(images[i], encoded_sentence)
                D_proba_list.append(probas)

            step_G_loss = generator_loss(D_proba_list, real_labels)
            validation_metrics['G_loss'] += step_G_loss

            # Encode the resulting last generated image
            encoded_images_subregions, encoded_image = self.image_encoder(generated_images[-1])
            
            # DAMSM loss for words and generated image
            all_damsm_losses = damsm_loss(encoded_sentence, encoded_image,
                                            encoded_words, encoded_images_subregions,
                                            captions_length, minibatch_indices, self.gammas)
            step_damsm_loss, step_word_loss, step_sent_loss = all_damsm_losses
            validation_metrics['damsm_loss'] += step_damsm_loss
            validation_metrics['word_loss'] += step_word_loss
            validation_metrics['sentence_loss'] += step_sent_loss
            validation_metrics['total_loss'] += (step_G_loss + self.Lambda * step_damsm_loss)

        logger.info("""Epoch [{}/{}] Validation Losses:
                       Discriminator: {:.4f},. Generator: {:.4f}, Damsm: {:.4f}.
                       """.format(epoch, self.num_epochs, 
                                validation_metrics['D_loss'].item(),
                                validation_metrics['G_loss'].item(),
                                validation_metrics['damsm_loss'].item()))
        return validation_metrics

    def change_models_mode(self, mode='train'):
        if mode == 'train':
            self.conditional_augmentation.train()
            self.generator.train()
            for discriminator in self.discriminators:
                discriminator.train()
        elif mode == 'eval':
            self.conditional_augmentation.eval()
            self.generator.eval()
            for discriminator in self.discriminators:
                discriminator.eval()
        else:
            raise NotImplementedError("Invalid mode, excpeted 'train' or 'eval', received {}".format(mode))

    def save_attngan(self, epoch, G_optimzer, D_optimizers, train_loss, val_loss, path_to_params):
        """Saves all relevant information about the attngan training process"""
        models = {
            'text_encoder': self.text_encoder.state_dict(),
            'conditional_augmentation': self.conditional_augmentation.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminators': [d.state_dict() for d in self.discriminators],
            'image_encoder': self.image_encoder.state_dict()
        }
        optimizers = {
            'G_optimizer': G_optimzer.state_dict(),
            'D_optimizers': [d_opt.state_dict() for d_opt in D_optimizers]
        }
        losses = {
            'train_loss': train_loss,
            'validation_loss': val_loss
        }
        torch.save({ "epoch": epoch,
                     "models": models,
                     "optimizers": optimizers,
                     "losses": losses
                   }, path_to_params)
        
    @staticmethod
    def prepare_real_fake_labels(batch_size, real_value=1, fake_value=0):
        """Given a batch size, will return a vector of shape [batch_size, 1] 
        filled with the given real_value and fake_value
        real_value default to 1, according to the paper loss function
        fake_value default to 0, according to the paper loss function"""
        real_labels = (torch.ones(batch_size) * real_value).to(device)
        fake_labels = (torch.ones(batch_size) * fake_value).to(device)
        return real_labels, fake_labels

    @staticmethod
    def prepare_label_smoothing(real_labels, fake_labels, smooth=0.1):
        """Implements label smoothing for the descriminator labels
            Described in section 3.4 of the Imroved Techniques for Training GANS (Point 6 on Video)
            Parameters:
                * real_labels (Tensor, shape: [batch_size]) vector filled with 1's
                * fake_labels (Tensor, shape: [batch_size]) vector filled with 0's
                * smooth (float, default=0.1): value to smooth labels with
        """
        smoothed_real = real_labels - smooth * torch.rand_like(real_labels)
        smoothed_fake = fake_labels + smooth * torch.rand_like(fake_labels)
        return smoothed_real, smoothed_fake

    def add_noise_to_images(self, images):
        """Hack specified in facebook AI improving GAN training talk """
        noisy_images = []
        for image in images:
            noisy_images.append(0.02 * torch.randn_like(image) * self.D_noise_decay + image)
        self.D_noise_decay *= 0.9999995
        return noisy_images

    @staticmethod
    def change_requires_grad(models_list, require_grad=True):
        for model in models_list:
            for param in model.parameters():
                param.require_grad = require_grad
    
    def tensorboard_log_scalers(self, epoch, set_type, metrics):
        self.tensorboard_writer.add_scalar('{}/D_loss'.format(set_type), metrics['D_loss'], global_step=epoch)
        self.tensorboard_writer.add_scalar('{}/G_loss'.format(set_type), metrics['G_loss'], global_step=epoch)
        self.tensorboard_writer.add_scalar('{}/word_loss'.format(set_type), metrics['word_loss'], global_step=epoch)
        self.tensorboard_writer.add_scalar('{}/sentence_loss'.format(set_type), metrics['sentence_loss'], global_step=epoch)
        self.tensorboard_writer.add_scalar('{}/damsm_loss'.format(set_type), metrics['damsm_loss'], global_step=epoch)
        self.tensorboard_writer.add_scalar('{}/total_loss'.format(set_type), metrics['total_loss'], global_step=epoch)

    @torch.no_grad()
    def tensorboard_log_generated_images(self, epoch, fixed_noise, train_sample, valid_sample):
        """this function will generate and log images"""
        def _generate_image(sample):
            # Generate an image based on training caption
            _, caption, caption_len, label, image_id = BirdsDataset.prepare_batch_date(sample)
            caption, caption_len = caption.unsqueeze(0), caption_len.unsqueeze(0)
            encoded_words, (encoded_sentence, _) = self.text_encoder(caption, caption_len)
            words_mask = caption == self.str2idx['<eos>']
            words_mask = words_mask[:, :encoded_words.size(2)]
            conditioned_vec, _, _ = self.conditional_augmentation(encoded_sentence)
            gan_input = torch.cat([conditioned_vec, fixed_noise], dim=1)  # the concatentated conditioning with the noise
            generated_images, _ = self.generator(gan_input, encoded_words, words_mask)
            final_g_images = []
            for i, image in enumerate(generated_images):
                image = image.squeeze(0)
                image = ((image + 1.0) * 127.5).type(torch.uint8)
                final_g_images.append(image)
            return final_g_images, image_id

        def _log_generated_images(generated_images, img_type, image_id):
            # Log all generated image sizes
            for i, image in enumerate(generated_images):
                img_size = 64 * (2 ** i) # 64, 64 * 2 = 128, 64 * 4 = 256...
                self.tensorboard_writer.add_image('{}_{}/generated_image_{}'.format(img_type, img_size, image_id), image, global_step=epoch)

        train_generated_images, train_image_id = _generate_image(train_sample)
        valid_generated_images, valid_image_id = _generate_image(valid_sample)

        _log_generated_images(train_generated_images, 'train', train_image_id)
        _log_generated_images(valid_generated_images, 'validation', valid_image_id)

    @torch.no_grad()
    def generate_new_image(self, epoch, step=None):
        self.generator.eval()
        sample = self.train_dataloader.dataset.draw_random_sample()
        captions = sample.caption.unsqueeze(0).to(device)
        captions_length = sample.caption_length.unsqueeze(0).to(device)
        img_id = sample.image_id.item()
        # Print a train single image
        encoded_words, (encoded_sentence, _) = self.text_encoder(captions, captions_length)
        words_mask = captions == self.str2idx['<eos>']
        words_mask = words_mask[:, :encoded_words.size(2)]
        conditioned_vec, mu, logvar = self.conditional_augmentation(encoded_sentence)
        z = torch.randn(conditioned_vec.shape[0], self.condition_dim).to(device)  # Noise vector
        gan_input = torch.cat([conditioned_vec, z], dim=1)  # the concatentated conditioning with the noise
        generated_images, _ = self.generator(gan_input, encoded_words, words_mask)
        eos_idx = self.train_dataloader.dataset.token2idx("<eos>")
        words = [self.train_dataloader.dataset.idx2token(idx) for idx in sample.caption if  idx != eos_idx]
        sentence = " ".join(words)
        plt.figure(figsize=(15,5))
        plt.suptitle("{}".format(sentence), fontsize=18)
        for i, image in enumerate(generated_images):
            image = image.squeeze(0).permute(1,2,0).contiguous().to('cpu').detach().numpy()
            image = ((image + 1.0) * 127.5).astype(np.uint8)
            size = self.images_sizes_list[i]
            plt.subplot(1,3,i+1)
            plt.imshow(image)
            plt.title("Level {}: {}x{} size image".format(i, size, size))
            plt.grid(False)
        if step:
            plt.savefig("/home/user_2/AttnGAN/Matan/AttnGAN/figures/generated_images/steps/epoch_{}_step_{}_id_{}.png".format(epoch, step, img_id))
        else:
            plt.savefig("/home/user_2/AttnGAN/Matan/AttnGAN/figures/generated_images/epoch_{}_id_{}.png".format(epoch, img_id))
        self.generator.train()


# This will be added as a loss to the conditional augmentation to learn to resemble the same 
# probability distribution of the sentence embedding
# ** Copy pasted from original code **
def KL_loss(mu, logvar):
    # -0.5 * sum(  1 + log(sigma^2) - mu^2 - sigma^2   )
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD