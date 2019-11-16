import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextEncoder(nn.Module):
    """TextEncoder will implement the TextEncoder according to the AttnGAN paper.
    Default values will be according to the values used in the paper
    """
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256,
                 num_layers=1, dropout_proba=0.5, bidirectional=True, init_value=0.1):
        """TextEncoder init function.
        Initialize a simple RNN encoder as in SEQ2SEQ models.
        Parameters:

        vocab_size (int): size of the vocabulary of the dataset used
        embedding_dim (int, default=300): size of the vector embedding dimension
        hidden_dim (int, default=256): The dimension of the hidden layer.
            Maintains the same dimension if bidirectional or not.
            If bidirectional, each cell hidden dim is hidden_dim/2, and the total hidden is concatenated.
            The hidden_dim is the word_features dimension that will be used all over the network.
            Denoted with `D` in the paper
        num_layers (int): number of LSTM layers of the encoder. Using only the output of the final layer
        dropout_proba (float, default=0.5, in (0,1]): the percent of the dropout that will be used after the 
                                        embedding layer and before the LSTM layer
        bidirectional (bool, default=True): boolean indicating if the LSTM should be bidirectional
        init_value  (float, default=0.1): value that will indicate the edges of the
                                          uniform initialization of the embedding layer

        TODO --->
        1. Check out differnet initialization for the weights (Xavier, Kaiming?)
        2. Check out if sorting the sequences by length improves computation time.
        """
        super(TextEncoder, self).__init__()

        # Vocabulary
        # self.vocabulary = vocabulary
        self.vocab_size = vocab_size

        # Dimensions of RNN
        # Maintain a deterministic dimension of hidden layer
        # (e.g. always 256, hidden_dim=256 if one directional or hidden_dim=256/2 dim for each cell which concatenated to 256)
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.num_directions = (2 if self.bidirectional else 1)
        self.hidden_dim = hidden_dim // self.num_directions
        self.num_layers = num_layers
        self.semantic_space_dim = hidden_dim  # The actual hidden dim, denoted as `D` in the paper

        # Constants
        self.dropout_proba = dropout_proba
        self.init_value = init_value

        # Define the Encoder layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_proba)
        self.RNN = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                           num_layers=self.num_layers, bidirectional=self.bidirectional,
                           batch_first=True)

        # Initialize the Encoder layers uniformly
        nn.init.uniform_(self.embedding.weight, -init_value, init_value)
        # The LSTM cell already have initialized weights to uniform weights.

    def forward(self, sentences, sentences_lengths):
        """ Forward pass of the text encoder.
        Parameters:
            sentences (tensor): padded tensor of integers that represent the tokens of all captions in the batch
                shape: (batch_size, max_sentence_len)
            sentences_lengths (tensor): 1D tensor containing the effective length of the given caption
                (effective_length is the length before the padding of <eos> tokens)
                shape: (batch_size)

        Returns:
            output (tensor): Tensor containing the output features from the last layer of the LSTM
                for each t. The output is a padded tensor with pad_value default to 0.
                shape: (batch_size, semantic_space_dim, seq_len):
            h_n (tensor): Tensor containing the hidden state for t=sentence_length
                shape: (batch_size, semantic_space_dim)
            c_n (tensor): Tensor containing th cell state for t=sentence_length
                shape: (num_layers * num_directions, batch_size, hidden_dim)

        Notes:
            * h_n and c_n are not effected by the batch_first parameter to the RNN.
            * The shapes of the encoder are as output by the RNN module, 
                thus need to do some conversions after using this module to be compatible with the paper
            """
        # Embedding and dropout
        embedded = self.embedding(sentences)
        embedded = self.dropout(embedded)

        # RNNs cells expect to get PackedSequence object
        packed_embedded = pack_padded_sequence(embedded, sentences_lengths, batch_first=True, enforce_sorted=False)

        # RNN forward pass
        packed_output, (h_n, c_n) = self.RNN(packed_embedded)

        # RNNs returns a PackedSequence object, this is the inverse of that operation.
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Batch first, and fix shapes
        output, h_n = self.handle_output_shapes(output, h_n, output.shape[0])

        # Finish
        return output, (h_n, c_n)

    @staticmethod
    def handle_output_shapes(encoded_words, encoded_sentence, batch_size):
        """Will handle the output of the text_encoder shapes.
            Fixes moves the batch axis to be batch_first.
            Concatenate the two layers if they exist
        """
        encoded_words = encoded_words.permute(0, 2, 1)  # (B, D, T); denoted as `e` in the paper
        encoded_sentence = encoded_sentence.permute(1, 0, 2).contiguous()
        encoded_sentence = encoded_sentence.view(batch_size, -1) # (B, D); denoted as `e‾`
        return encoded_words, encoded_sentence
