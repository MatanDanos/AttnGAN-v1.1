import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision.models import inception_v3
from loguru import logger


class ImageEncoder(nn.Module):
    """ Image Encoder at the end of the generation stages.
    The image encoder is based on the learned features of the inception-v3 network.

    Parameters:
        word_feature_dim (int): the dimension of the word features of the network. 
                                 * Denoted with D in the paper.

    TODO -->
    1. Add different initialization to the linear layers? (kaiming, xavier?)
    """
    def __init__(self, word_feature_dim=256):
        super(ImageEncoder, self).__init__()
        logger.debug("Started loading the Inception-v3 model")
        inception = inception_v3(pretrained=True, progress=False)
        logger.debug("Finished loading the Inception-v3 model")

        self.word_feature_dim = word_feature_dim
        self.semantic_space_dim = word_feature_dim

        # Freeze all layers of pre-trained network for fast computation
        for param in inception.parameters():
            param.requires_grad = False

        # First layers in the forward step:
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a

        # Second step layers of the forward pass
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c

        # Pooling layers
        self.max_pool2d = nn.MaxPool2d(3, stride=2)
        self.avg_pool2d = nn.AvgPool2d(8)  # Last pooling layer..

        # Two trainable fully connected layers
        # They are used to convert the image feature to a common semantic space of the text features
        self.fc_local = nn.Conv2d(768, word_feature_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc_global = nn.Linear(2048, word_feature_dim)

        self.init_range = 0.1  # TODO get from config
        init.uniform_(self.fc_local.weight, a=-self.init_range, b=self.init_range)
        init.uniform_(self.fc_global.weight, a=-self.init_range, b=self.init_range)

    def forward(self, image):
        """Forward pass through the image encoder Inception-V3 pretrained on ImageNet.
        Steps of the forward pass:
        1. Rescale the input image to 299x299 pixels (consistency with inception-v3 model)
        2. Pass the image through the layers until "mixed_6e" layer where we use the output as
            local feature matrix
        3. Continue passing the image through the layers until the last average pooling layer
            in order to extract the global feature vector
        4. Pass the image features to a common semantic space of text features by adding a perceptron layer

        Parameters:
            image (tensor, shape: [batch_size, C, H, W]): Tensor representing a batch of images
                Usaully this will be the output of the last generator, thus [*, 3, 256,256]

        Output:
            local_features (tensor, shape: [batch_size, word_feature_dim, 289]): the local feature matrix
            global_feature (tensor, shape: [batch_size, word_feature_dim]): the global feature vector

        TODO -->
        1. try a "bicubic" interpolation mode for better results? (they used bilinear)
        """
        batch_size = image.shape[0]
        image = F.interpolate(image, size=(299, 299), mode='bilinear', align_corners=False)  # 299 x 299 x 3

        # First step, to produce the local feature matrix
        image = self.Conv2d_1a_3x3(image)  # 32 x 149 x 149
        image = self.Conv2d_2a_3x3(image)  # 32 x 147 x 147
        image = self.Conv2d_2b_3x3(image)  # 64 x 147 x 147
        image = self.max_pool2d(image)  # 64 x 73 x 73
        image = self.Conv2d_3b_1x1(image)  # 80 x 73 x 73
        image = self.Conv2d_4a_3x3(image)  # 192 x 71 x 71
        image = self.max_pool2d(image)  # 192 x35 x 35
        image = self.Mixed_5b(image)  # 256 x 35 x 35
        image = self.Mixed_5c(image)  # 288 x 35 x 35
        image = self.Mixed_5d(image)  # 288 x 35 x 35
        image = self.Mixed_6a(image)  # 768 x 17 x 17


        # Second step, to produce the global feature vector
        image = self.Mixed_6b(image)  # 768 x 17 x 17
        image = self.Mixed_6c(image)  # 768 x 17 x 17
        image = self.Mixed_6d(image)  # 768 x 17 x 17
        image = self.Mixed_6e(image)  # 768 x 17 x 17

        # Save the local feature matrix
        local_features = image.detach()

        image = self.Mixed_7a(image)  # 1280 x 8 x 8
        image = self.Mixed_7b(image)  # 2048 x 8 x 8
        image = self.Mixed_7c(image)  # 2048 x 8 x 8
        image = self.avg_pool2d(image)  # 2048 1 x 1

        # Save the global feature vector
        global_feature = image.view(batch_size, -1)

        # Converting to a common semantic space with the word features
        ## local_featuers are transposed twice so linear layer will be able to receive a matrix,
        ## and perform the operations on the columns
        local_features = self.fc_local(local_features).view(batch_size, self.word_feature_dim, -1)
        global_feature = self.fc_global(global_feature)

        return local_features, global_feature
