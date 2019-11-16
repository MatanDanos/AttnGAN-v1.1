import torch
import torch.nn as nn

from loguru import logger


class Attention(nn.Module):
    """The attention mechanism as implemented in the paper
    Parameters:
        word_feature_dim (int, default=256): the dimension of the word encoding as output by the text_encoder
        image_feature_dim (int, default=128): the dimension of the image features, equivalent to channels

    Notation in the paper for variables in the code:
        * D => word_feature_dim
        * D\hat => image_feature_dim
        * T => seq_length
        * N => H * W
        * e => word_features
        * h => image_features
        * U => self.fc
        * s_j,i => S[j,i]
        * 1 <= j <= N, 1 <= i <= T
        * beta_j,i => attn_coefficients[j,i]
        * F_attn(e,h) = (C_0,...,C_N-1) => attn   (c for context vector)

    Masking:
        The idea: masking irrelevant encoding of <sos> tokens.
        We need the masking so the attention will not attend to words that did not appear in the 
        sentence, but the shape allows it to. This is because all sentences were padded into 
        a max_len padding to have tensor shape consistency.
        To make this work, we use the mask to zero out all of the attn_coeficents by
        setting the number before the softmax to -infinity. This will make these zero for later
        multiplication with the encoded words.

    Note: 
        I think their implementation for the masking is wrong.
        since the data from a each sample in the batch will contaminate 
        the info in all other samples in the batch.
        i.e. the sequence length of a single sample affects all other samples and might erase 
        the attention coefficient for a word that might exist!
        This shouldn't happen. Thus changed. This required simply different reshaping.

    TODO -->
    1. Use different patching (not just row stack of the rows, but try 8x8 patchs) as image features
    """
    def __init__(self, word_feature_dim=256, image_feature_dim=128):
        super(Attention, self).__init__()
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.fc = nn.Linear(self.word_feature_dim, self.image_feature_dim)

    def forward(self, word_features, image_features, words_mask):
        """
        Parameters:
            word_features (tensor, shape: [batch_size, word_feature_dim, seq_length]):
                Feature vector for each word in the sentence, as output by text_encoder
            image_features (tensor, shape: [batch_size, image_feature_dim, H, W]):
                The image representation as a feature 3D array (e.g., [*,128,64,64])
                Recieved as output from the generator F_i stage
            words_mask (tensor, shape: [batch_size, seq_len]):
                Tensor of uint8 that holds 0 or 1 according if the encoding should be masked during
                the attention process because it represents <sos> token.

        Returns:
            attn (tensor, shape: [B, image_feature_dim, H * W]) The attention matrix
            attn_coefficients (tensor, shape: [B, H * W, seq_len]) The attention coefficients
        """
        # Compute dimensions
        batch_size = word_features.shape[0]
        
        ## Image dimensions
        image_feature_dim = image_features.shape[1]
        height, width = image_features.shape[2], image_features.shape[3]
        N = height * width  # Number of subregions of the image_features (just columns of the image) == 4096 for 64x64 image

        ## Word dimensions
        word_feature_dim = word_features.shape[1]
        seq_len = word_features.shape[2]

        # Changing dimensions, reshaping and transposing to prepare for matrix multiplication
        values = self.fc(word_features.transpose(1, 2)).transpose(1, 2)        
        queries = image_features.view(batch_size, image_feature_dim, N).transpose(1, 2)

        # Compute the attention coefficients using basic torch operations
        S = torch.bmm(queries, values)  # Cell S[j,i] = s_i,j in the paper, shape: [B, image_feature_dim*image_feature_dim, seq_len])

        # The masking, will put -inf in the columns so it will be zerod out by the exponent
        words_mask = words_mask.unsqueeze(1)
        S.masked_fill_(words_mask.detach(), -float('inf'))

        # Manual softmax to avoid NANs due to the -inf
        S_exp = torch.exp(S)
        attn_coefficients = torch.div(S_exp, torch.sum(S_exp, dim=2, keepdim=True))

        # Compute the final attention vector attn=(C_0,...C_N-1) by a single matrix multiplication
        attn = torch.bmm(values, attn_coefficients.transpose(1, 2))

        return attn, attn_coefficients
