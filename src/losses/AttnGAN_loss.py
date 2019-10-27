# This file will implement the AttnGAN loss functions.
# As described in the paper in section 3.1 `Attentional Generative Network`
import torch
import torch.nn.functional as F
from loguru import logger

def generator_loss(D_probabilities, target_labels):
    """Implements the generator loss function as decribed in the paper.
       Denoted in the paper as `L_G= \sum L_G_i`
       Parameters:
            D_probabilities (list of tensors, shape: [B, 2]):
                    The i'th tensor represents the probabilities
                    given by the i'th discriminator to the generatated images.
                        where in the proba[:, 0] is the conditional probabilities
                        where in the proba[:, 1] is the unconditional probabilities
            target_labels (tensor, shape: [B]):
                    tensor containing 1 for real labels.
                    Possible Confusion: this is 1 because for the generator,
                    if the images are real he wants to make the discriminator
                    predict 1, thus his target label is 1 for real.
        Returns:
            total_loss according to the paper defined generator loss
    """
    total_loss = 0
    for i, probas in enumerate(D_probabilities):
        if torch.any(torch.isnan(probas)):
            probas = torch.where(torch.isnan(probas), 1e-5 * torch.ones_like(probas), probas)
        if torch.any(torch.isinf(probas)):
            probas = torch.where(torch.isinf(probas), 1e-5 * torch.ones_like(probas), probas)
        cond_loss = F.binary_cross_entropy(probas[:, 0], target_labels)
        uncond_loss = F.binary_cross_entropy(probas[:, 1], target_labels)
        total_loss += (cond_loss + uncond_loss) / 2    # They didn't use /2 in their code, but yes in the paper
    return total_loss


def discriminator_loss(real_probabilities, fake_probabilities, wrong_probabilities, real_labels, fake_labels):
    """
        Implements the discriminator loss function as decribed in the paper.
        Parameters:
            real_probabilities (tensor, shape: [B, 2]): tensor representing the probabilities
                            given by the discriminator to the real images.
                            where in the proba[:, 0] is the conditional probabilities
                            where in the proba[:, 1] is the unconditional probabilities
            fake_probabilities (tensor, shape: [B, 2]): tensor representing the probabilities
                            given by the discriminator to the fake images.
                            where in the proba[:, 0] is the conditional probabilities
                            where in the proba[:, 1] is the unconditional probabilities
            wrong_probabilities (tensor, shape: [B, 2]): tensor representing the probabilities
                                given by the discriminator to real images with wrong captions
                            where in the proba[:, 0] is the conditional probabilities
                            * proba[:, 1] are irrelevant numbers
                            * Were not mentioned in the paper but appeared in the code
            real_labels (tensor, shape: [B]): tensor containing 1 for real labels.
            fake_labels (tensor, shape: [B]): tensor containing 0 for fake labels.
        Returns:
            total_loss according to the paper defined discriminator loss
    """
    def remove_inf_nan_probas(probabilities):
        if torch.any(torch.isnan(probabilities)):
            probabilities = torch.where(torch.isnan(probabilities), 1e-5 * torch.ones_like(probabilities), probabilities)
        if torch.any(torch.isinf(probabilities)):
            probabilities = torch.where(torch.isinf(probabilities), 1e-5 * torch.ones_like(probabilities), probabilities)
        return probabilities

    # Conditional error
    real_cond_error = F.binary_cross_entropy(remove_inf_nan_probas(real_probabilities[:, 0]), real_labels)
    fake_cond_error = F.binary_cross_entropy(remove_inf_nan_probas(fake_probabilities[:, 0]), fake_labels)
    wrong_cond_error = F.binary_cross_entropy(remove_inf_nan_probas(wrong_probabilities[:, 0]), fake_labels[:-1])

    # Unconditional error
    real_uncond_error = F.binary_cross_entropy(remove_inf_nan_probas(real_probabilities[:, 1]), real_labels)
    fake_uncond_error = F.binary_cross_entropy(remove_inf_nan_probas(fake_probabilities[:, 1]), fake_labels)

    unconditional_loss = (real_uncond_error + fake_uncond_error) / 2
    conditional_loss = (real_cond_error + fake_cond_error + wrong_cond_error) / 3 # This is different in their loss..
    total_loss = unconditional_loss + conditional_loss
    return total_loss


def feature_matching(real_D_feature, fake_D_feature):
    """Implements the feature matching loss as described in Improving GAN training paper
    in section 3.1.

    Can be used as a regularization factor to the G loss.

    Parameters:
        real_D_feature (tensor, shape: (B, c, h, w)):
                    the intermediate layer of the discriminator on a batch of real images

        fake_D_feature (tensor, shape: (B, c, h, w)):
                    the intermediate layer of the discriminator on a batch of fake images
    Return:
        loss value, which is the l2 norm of the distance between these two features
        `||Ex~p_data f(x) - Ez~p_z f(G(z))||_2^2`
    """
    real_features_mean = torch.mean(real_D_feature, dim=0)
    fake_features_mean = torch.mean(fake_D_feature, dim=0)
    return torch.norm(real_features_mean - fake_features_mean, p=2)
