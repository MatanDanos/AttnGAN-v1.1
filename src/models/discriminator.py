import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    This is a simple discriminator network
    In initialization of the discriminator, one can choose the size it will accept.
    
    During the training processes, one will have to create a discriminator for each one of the 
    generators. Thus each discriminator will have to accept different input sizes.

    Note: In the paper, they described the process of using a simple conditional GAN.
    That's why we chose to implement only a conditional GAN and not a a DCGAN.
    Thus this implementation should be the same as the implementation of their conditional GAN.

    Predicting using a discriminator works as follows:
        1. Creates a feature map from the image, which is of shape 8*ndf x 4 x 4
            Denoted as D_feature_map
        2. Using the above feature map, we reduce to a single output.
            We return in the forward function, both the conditiona and unconditional
            prediction of the discriminator. 
    """
    def __init__(self, input_size, num_D_filters=64, semantic_dim_size=256):
        """Discriminator Initialization Function
        Parameters:
            input_size (int): The shape of the input image
            num_D_filters (int, default=64): Number discriminator filters in the first layer of the discriminator
            semantic_dim_size (int, default=256): The semantic space dimension size (encoded words dimension)

        Notes:
            1. This init method might seem complex, but it follows their rules for 
                discriminators of higher levels and not just 256x256 pixels input.
                This will also be usefull in the forward pass, we can just iterate the module list
                and thus avoid if-then syntax.
        """
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.input_shape = (3, input_size, input_size)
        self.num_D_filters = num_D_filters  # Also denoted in comments as ndf
        self.semantic_dim_size = semantic_dim_size # denoted in comments as ed (encoded dimension)

        # Create the a list of downsample and rescaling layers that creates the features map
        # if input is iterated one by one
        self.layers = nn.ModuleList()

        # First layer: (3 x in_size x in_size) --> (ndf x in_size/16 x in_size/16)
        self.layers.append(self._get_downsampler(3, self.num_D_filters, 16))

        # Next layers - in every iteration:
        #   1. width and height downscale by 2
        #   2. channels upscale by 2
        num_filters = self.num_D_filters * 8
        current_shape = self.input_size // 16    # 64/16 == 4, 128/16 == 8, 256/16 == 16...
        while current_shape != 4:
            self.layers.append(self._get_downsampler(num_filters, num_filters * 2, 2))
            current_shape = current_shape // 2
            num_filters = num_filters * 2
        
        # Next layers - in every iteration:
        #   1. width and height stay the same,
        #   2. channels upscale by 2 every iteration
        current_shape = self.input_size // 16    # 64/16 == 4, 128/16 == 8, 256/16 == 16
        while current_shape != 4:
            self.layers.append(self._get_scale_invariant_conv_layer(num_filters, num_filters // 2))
            num_filters = num_filters // 2
            current_shape = current_shape // 2

        # This is the convolution that will take into account the condition vector
        # Should be used on the feature_map and the condition to prepare the conditional feature map
        # for the final layer
        self.conv_features_conditions = self._get_scale_invariant_conv_layer(self.num_D_filters * 8 +
                                                                             self.semantic_dim_size,
                                                                             self.num_D_filters * 8)

        # This is the final layer that will output the results
        self.final = nn.Sequential(nn.Conv2d(self.num_D_filters * 8, 1, kernel_size=4, stride=4),
                                   nn.Sigmoid())

    def forward(self, images, conditions):
        """The forward pass of the discriminator
        Parameters:
            images (tensor, shape: (B, 3, input_size, input_size)):
                    the batch of images we want to discriminate
            conditions (tensor, shape: (B, semantic_dim_size)):
                    the encoded sentence as output by the text encoder
        Returns:
            tuple (conditional_probability, unconditional_probability)
        """
        # Run through all rescaling layers to produce the images featuresmap
        x = images
        for layer in self.layers:
            x = layer(x)
        features_map = x.detach()

        # Conditional D result
        joint_features_condition = self._concatenate_features_conditons(self.semantic_dim_size, x, conditions)
        convolved_features_condition = self.conv_features_conditions(joint_features_condition)
        conditonal_probability = self.final(convolved_features_condition).view(-1, 1)

        # Unconditional D result
        unconditonal_probability = self.final(x).view(-1, 1)

        return torch.cat((conditonal_probability, unconditonal_probability), 1), features_map

    @staticmethod
    def _get_downsampler(in_filters, out_filters_first, scale_factor):
        """Creates a downsampler sequential neural block.
        the block is built of several Conv2D-BatchNorm-LeakyReLU layers.
        The first layer does not have batchnorm, unless the the scale_factor is 2.
        Note: 
            The downsample changes the height and width by the scale factor, 
            but the number of channels of the output will be:
                (out_filters_first * 2**(log(scale_factor)-1)).
                Examples:
                    For scale_factor = 16 there will be `out_filters_first * 8` channels in the output
                    For scale_factor = 2 there will be `out_filters_first` in the output
        Parameters:
            in_filters (int, default=3): the number of filters/channels of the input.
            out_filters_first (int, default=64): The number of filters that should be in the output
                                                 of the first layer in the block
                                                 (usually denoted as ndf)
            scale_factor (int, default=16): how much does the returned block should scale the image
                                            Note: This number must be a power of 2!
        Returns:
            The sequential pytorch model.

        """
        if scale_factor == 2:
            return nn.Sequential(nn.Conv2d(in_filters, out_filters_first, 4, 2, 1, bias=False),
                                 nn.BatchNorm2d(out_filters_first),
                                 nn.LeakyReLU(0.2, inplace=True)
                                )
        # Initialize hyperparameters for dimensions
        current_scale = scale_factor
        in_filters_next = in_filters
        out_filters_next = out_filters_first
        block_layers = []
        while current_scale != 1:
            block_layers.append(nn.Conv2d(in_filters_next, out_filters_next, 4, 2, 1, bias=False))
            if current_scale != scale_factor:
                block_layers.append(nn.BatchNorm2d(out_filters_next))
            block_layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Update dimensions for next layer
            current_scale = current_scale // 2
            in_filters_next = out_filters_next
            out_filters_next = out_filters_next * 2
        return nn.Sequential(*block_layers)

    @staticmethod
    def _get_scale_invariant_conv_layer(in_filters, out_filters):
        """This method will return a sequential conv layer that does not change
        the height and width of the image (images scale)
        The number of input and output filters/channels are controlled as parameters
        
        Parameters:
            in_filters (int): number of filters/channels of the input image
            oyt_filters (int): number of filters/channels of the output 
        
        Return: 
            A sequential pytorch model, consist of a single Conv2D layer, Batch normalization and leakyRELU
        """
        return nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_filters),
            nn.LeakyReLU(0.2, inplace=True)
        )

    @staticmethod
    def _concatenate_features_conditons(nef, features, conditions):
        conditions_reshape = conditions.view(-1, nef, 1, 1)   # shape: (B, 8*ndf x 4 x 4)
        conditions_repeat = conditions_reshape.repeat(1,1,4,4)  # Shape: (B x (8*ndf+ed) x 4 x 4)
        joint_features_condition = torch.cat((features, conditions_repeat), 1) # Shape: (B x 8*(ndf) x 4 x 4)
        return joint_features_condition
