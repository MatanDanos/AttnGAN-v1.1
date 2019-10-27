# This file will implement the DAMSM loss functions.
# As described in the paper in section 3.2 `Deep Attentional Multimodal Similarity Model`
import torch

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sentence_loss(global_sentence_vec, global_image_vec, minibatch_indices, gamma3):
    """This is the sentence loss. 
    Loss is computed between the entire sentence encoding and the entire image encoding.
    Represent the match between the entire sentence and entire image.
    
    Parameters:
        global_sentence_vec - (tensor, shape: (B, D)):
                    Tensor representing the entire encoded sentence.
                    Denoted as `e‾` in the paper.
        global_image_vec - (tensor, shape: (B, D)):
                    Tensor representing the entire encoded image.
                    Denoted as `v‾` in the paper.
        minibatch_indices - (tensor, shape (B)):
                    Tensor of indices [0:B-1] that will be used for the minibatch mismatching 
                    of the samples in the last step of the loss function.
        gamma3 - (float): a number that helps smoothes things out
    Returns:
        loss1, loss2 - (tensor, tensor): two 1x1 tensors that represents the loss;
                        Denoted as L1s, L2s in the paper
    
    Important Note:
        The loss function defined in the paper has to be computed manually,
        since there is no pairwise distance for cosine similarity.
        This is because the R_matrix should have a cosine similarity between every pair 
        of sample in the batch.
        See feature request: https://github.com/pytorch/pytorch/issues/11202
    """
    batch_size = global_image_vec.shape[0]

    # Compute the cosine similarity for every pair of samples in the batch manually:
    # i.e compute: (e * v^T) / ||e|| * ||v|| , which is eq (10) R(Q,D) for sentences (page 1320 in the paper)
    # Will leverage torch matrix multiplication and outer product (ger)
    # ! Important: the order of the inputs to the functions matter!!!
    numerator = torch.mm(global_sentence_vec, global_image_vec.transpose(0, 1))  # e * v^T --> shape = BxB
    global_sentence_norm = torch.norm(global_sentence_vec, dim=1) # ||e|| --> shape = B
    global_image_norm = torch.norm(global_image_vec, dim=1) # ||v|| --> shape = B
    denominator = torch.ger(global_sentence_norm, global_image_norm).clamp(min=1e-8) # ||e|| * ||v|| ^T --> shape = (BxB)
    R = gamma3 * (numerator / denominator)  # Final pairwise cosine similarity, Shape: BxB

    # Calculating the loss
    # ! Huge trick here:
    # We use range as labels to the cross entropy loss function
    # Instead of just doing a loop that will sum the individual losses
    loss1 = F.cross_entropy(R, minibatch_indices)
    loss2 = F.cross_entropy(R.transpose(0, 1), minibatch_indices)

    return loss1, loss2


def words_loss(words_feature_matrix, visual_features_matrix, captions_length, minibatch_indices, gammas):
    """This is the words loss. 
    Loss is computed between the entire sentence encoding and the entire image encoding.
    Represent the match between the entire sentence and entire image.
    
    Parameters:
        words_feature_matrix - (tensor, shape: (B, D, T)):
                    Tensor representing the entire encoded sentence.
                    Each i'th column of a single sample represents the encoding of the i'th word
                        * B is the batch dimension
                        * D is the semantic dimension, 
                        * T is the number of maximum number of words in a sentence in the batch
                    Denoted as `e` in the paper.
        visual_features_matrix - (tensor, shape: (B, D, 289)):
                    Tensor representing the entire encoded image.
                    Each i'th column of a single sample represents the encoding of the i'th subregion in the image
                    Denoted as `v` in the paper.
                        * B is the batch dimension
                        * D is the semantic dimension,
        captions_length - (tensor, vector of size B):
                    Holds the length of the captions for every sample in the batch.
                    Excluding the <SOS> and <EOS> embeddings.
        minibatch_indices - (tensor, shape (B)):
                    Tensor of indices [0:B-1] that will be used for the minibatch mismatching 
                    of the samples in the last step of the loss function.
        gammas (dictionary): representes the loss function smoothing factors.
                    Accessing gamma_i with gammas['i'].
                    Denoted in the paper as gammas = {'1': gamma1, '2':gamma2, '3': gamma3}
    Returns:
        loss1, loss2 - (tensor, tensor): two 1x1 tensors that represents the loss;
                        Denoted as L1w, L2w in the paper

    # ! Important Notes:
        1. They originaly implemented this loss function in a for loop, for every sample in the batch.
        That is to avoid computing the similarity of unnecessary words
        because not all words are contained in the caption length (empty <EOS> encoding).
        Thus, we can save time.
        2. The alpha values can be used to draw the image later on.
    """
    batch_size = words_feature_matrix.shape[0]

    attn_maps = []
    matching_scores = []
    for sample_idx in range(batch_size):

        # Step 0: Names for the vectors for readability ("readability counts")
        # Note: Repeating the sample words B times in order to have an similarity score of the words
        # with every one of the images. Since the final loss function requires the interleaving similarity 
        # between different samples in the batch (see eq(11))
        sample_num_words = captions_length[sample_idx]
        sample_words = words_feature_matrix[sample_idx, :, :sample_num_words]
        sample_words = sample_words.repeat(batch_size, 1, 1)  # Shape: (B, D, num_words)

        # Step 1: normalized similarity as described in the paper (eq (7) & eq (8))
        similarity = torch.bmm(sample_words.transpose(1, 2), visual_features_matrix)  # Similarity. shape: (B, num_words, 289)
        normalized_similarity = F.softmax(similarity, dim=1)  # Dim to compute softmax is the words dimension. shape: (B, num_words, 289)

        # Step 2: Compute context vectors and attention (eq (9))
        # the context vector is a dynamic representation of the image subregions that relate to the words.
        # context[:, i] is the representation of the image's sub regions that relate to the i'th word in the sentence.
        alpha = F.softmax(gammas['1'] * normalized_similarity, dim=2) # Dim to compute softmax is the subregions dimension. shape: (B, num_words, 289)
        context = torch.bmm(visual_features_matrix, alpha.transpose(1, 2)) # Shape: BxDxT

        # Step 3: Attention driven image-text matchin score  (denoted as R(Q,D) in the paper eq (10))
        # 3.1 Cosine similarity between the visual context and the sentence (Denoted as R(c, e) in the paper)
        cosine_similarity_matrix = F.cosine_similarity(sample_words.transpose(1, 2), context.transpose(1, 2)) # shape: (B, D)
        # 3.2 Compute the matching score for the given sample text with every image in the batch
        R = torch.log(torch.sum(torch.exp(gammas['2'] * cosine_similarity_matrix), dim=1, keepdim=True))  # shape: (B,1)
        
        # Step 4: End of loop, keep results in a list
        attn_maps.append(alpha[sample_idx])  # alpha represents the attention (and will be needed later for visualization)
        matching_scores.append(R) # keep matching_score for each sample
    
    # Text-Image Matching score between every pair of samples in the batch:
    matching_scores = torch.cat(matching_scores, 1)  # Shape: (B,B)
    smoothed_matching_scores = gammas['3'] * matching_scores
    
    # Same trick as in the sentence loss to compute the final loss
    loss1 = F.cross_entropy(smoothed_matching_scores, minibatch_indices)
    loss2 = F.cross_entropy(smoothed_matching_scores.transpose(0, 1), minibatch_indices)

    return loss1, loss2


def damsm_loss(global_sentence_vec, global_image_vec,
               words_feature_matrix, visual_features_matrix,
               captions_length, minibatch_indices, gammas):
    """Computes the total damsm loss that is comprised of the sentence and words losses
    Return also the sentence and word loss values for logging purposes
    Parameters:
        Check out the parameters in word_loss and sentence_loss.
    Returns: 
        The sum of the word loss and sentence loss
    """
    word_loss1, word_loss2 = words_loss(words_feature_matrix, visual_features_matrix, captions_length, minibatch_indices, gammas)
    sent_loss1, sent_loss2 = sentence_loss(global_sentence_vec, global_image_vec, minibatch_indices, gammas['3'])

    batch_words_loss = word_loss1 + word_loss2
    batch_sent_loss = sent_loss1 + sent_loss2
    batch_damsm_loss = batch_words_loss + batch_sent_loss  # Final DAMSM loss
    return batch_damsm_loss, batch_words_loss, batch_sent_loss