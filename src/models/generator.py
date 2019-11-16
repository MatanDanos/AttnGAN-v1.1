import torch
import torch.nn as nn
import torch.nn.functional as F

# Project Modules
from models.attention import Attention


class UpsampleBlock(nn.Module):
    """ Upsample block doubles the H and W of the input with scale factor = 2
        then uses conv2d to reduce number of channels to out_dim
        we use out_dim * 2 because glu operation cuts the size in half
    """
    def __init__(self, in_dim, out_dim):
        super(UpsampleBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.upsample_block = nn.Sequential(
            nn.ConvTranspose2d(self.in_dim, self.in_dim, 3, stride=2, padding=1, output_padding=1), # Changed upsample to ConvTranspose2d to have gradients! From Tip 5 in the video.
            # nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.in_dim, self.out_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_dim * 2)
        )
    
    def forward(self, input_vector):
        output = self.upsample_block(input_vector)
        return F.glu(output, dim=1)


class GenaratorInitStage(nn.Module):
    """ Generator init stage
        takes the in_dim (sentence vector + conditional aug vector = 200)
        and enlarges to (out_dim(=128) * 16 = 2048) x 4 x 4 x 2 (the 2 is reduced after glu)
        Then upsampling is reducing the channels and enlarges the H & W to : out_dim x 64 x 64
    """
    def __init__(self, in_dim, out_dim):
        super(GenaratorInitStage, self).__init__()
        self.in_dim = in_dim # 200
        self.out_dim = out_dim * 16  # 128 * 16 = 2048
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.out_dim * 4 * 4 * 2)  # multiply by 2 for glu that divides it later
        )
        self.upsample1 = UpsampleBlock(self.out_dim, self.out_dim // 2)
        self.upsample2 = UpsampleBlock(self.out_dim // 2, self.out_dim // 4)
        self.upsample3 = UpsampleBlock(self.out_dim // 4, self.out_dim // 8)
        self.upsample4 = UpsampleBlock(self.out_dim // 8, self.out_dim // 16)

    def forward(self, z_c_vector):
        input_vector = z_c_vector
        out_code = F.glu(self.fc(input_vector), dim=1)
        out_code = out_code.view(-1, self.out_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        return out_code


class ResidualLayer(nn.Module):
    """ Residual layer 
        takes the input with n channels and pass it through
        a conv2d net with facotr 2 then glu (divides channels by 2)
        then again through a conv2d net.
        Adds the result to the input
        The result is the same size of the input

    """
    def __init__(self, channels):
        super(ResidualLayer, self).__init__()
        self.residual_layer_1 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2)
            #glu
        )
        self.residual_layer_2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels )
        )
    def forward(self, input):
        residual = input
        output = self.residual_layer_1(input)
        output = F.glu(output, dim=1)
        output = self.residual_layer_2(output)
        output += residual
        return output


class GeneratorResidualStage(nn.Module):
    """ Generator residual stage
        This stage has n=2 residual layers (a parameter) then an upsample (by factor 2)
        The result has same amount of channels = generator_features_number=128
        but H & W are larger by a factor of 2

    """
    def __init__(self, generator_features_number=128, num_of_residual_layers=2 ):
        super(GeneratorResidualStage, self).__init__()
        self.generator_features_number = generator_features_number
        self.num_of_residual_layers = num_of_residual_layers
    
        self.residual_layers_list = []
        for i in range(self.num_of_residual_layers):
            self.residual_layers_list.append(ResidualLayer(self.generator_features_number * 2))
        self.residual_layer = nn.Sequential(*self.residual_layers_list)
        self.upsample = UpsampleBlock(self.generator_features_number * 2, self.generator_features_number)

    def forward(self, h_c_vector):
        output = self.residual_layer(h_c_vector)
        return self.upsample(output)


class ImageGeneratorLayer(nn.Module):
    """ Image Generator layer is denoted by G0, G1, ... in AttnGAN
        reduces input channels from in_dim(=128) to 3 for RGB image
            Denoted in the diagram as the convolution block (orange block)
    """
    def __init__(self, in_dim):
        super(ImageGeneratorLayer, self).__init__()
        self.in_dim = in_dim
        self.image = nn.Sequential(
            nn.Conv2d(in_dim, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()  # Suggestion 1 for improvement in How To Train a GAN video
        )

    def forward(self, h_vector):
        return self.image(h_vector)


class GeneratorNetwork(nn.Module):
    """GeneratorNetwork implementation of the AttnGAN paper."""
    def __init__(self, input_vector_size=200, generator_features_number=128,
                       word_features_number=256, residual_stages_number=2,
                       residual_layers_in_stage_number=2):
        """ Init of Generator network
        Includes all the elements confined inside the red dotted line in the AttnGAN diagram.

        Parameters:

        input_vector_size - z + c concatenated vector
        generator_features_number - how many features will the generator have. # 128
        word_features_number - how many features does the words vector have. # 256
        residual_stages_number - how many residual stages in the model
        residual_layers_in_stage_number - how many layers in each residual stage

        Note: Conditional Augmentation is done outside in trainer
        """
        super(GeneratorNetwork, self).__init__()
        self.input_vector_size = input_vector_size
        self.generator_features_number = generator_features_number
        self.word_features_number = word_features_number
        self.residual_stages_number = residual_stages_number
        self.residual_layers_in_stage_number = residual_layers_in_stage_number
        
        init_stage = GenaratorInitStage(self.input_vector_size, self.generator_features_number) 
        get_img_stage = ImageGeneratorLayer(self.generator_features_number)

        self.stages_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.image_generators = nn.ModuleList()

        self.stages_layers.append(init_stage) #F0,F1..
        self.image_generators.append(get_img_stage)

        for i in range(self.residual_stages_number):
            self.stages_layers.append(GeneratorResidualStage(self.generator_features_number, self.residual_layers_in_stage_number))
            self.image_generators.append(ImageGeneratorLayer(self.generator_features_number))
            self.attention_layers.append(Attention(self.word_features_number, self.generator_features_number))

    def forward(self, input_vector, word_features, words_mask):
        """ forward of generator 
        Parameters:
            input_vector - the concatenated noise (z) and conditional agumentation (c) vector (size 200)
            word_features - the word features (encoded) used as input to Attn elements (size 256 x seq_length)
            words_mask (tensor, shape: [batch_size, seq_len]):
                Tensor of uint8 that holds 0 or 1 according if the encoding should be masked during
                the attention process because it represents <sos> token.
                Will be passed to the attention layers.
        Returns:
            images - list of fake images generated by the Generators (different sizes) 
                     used in discriminator and to calc losses
            att_maps - list of attention maps calculated
        """
        images = []
        att_maps = []  # TODO for what?
        init_stage = self.stages_layers[0]
        first_generator = self.image_generators[0]
        h_vector = init_stage(input_vector)    # B x 128 x 64 x 64
        images.append(first_generator(h_vector))
        batch_size = h_vector.size(0)

        for i in range(0, self.residual_stages_number):
            current_stage = self.stages_layers[i+1]
            current_attention = self.attention_layers[i]
            current_generator = self.image_generators[i+1]
            att, attn_coefficients = current_attention(word_features, h_vector, words_mask)    # att B x 128 x (64*64)
            att = att.view(batch_size, -1, h_vector.size(2), h_vector.size(3))
            h_c_vector = torch.cat((h_vector, att), 1) # concatenate h & c vectors
            h_vector = current_stage(h_c_vector)
            images.append(current_generator(h_vector))
            if att is not None:
                att_maps.append(att)

        return images, att_maps

