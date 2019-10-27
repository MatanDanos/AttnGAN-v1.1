# This file will implement images

import os
from skimage.io import imsave


def visualize_image(image, image_name):
    """Given an image, will save the image to the figures directory
    Parameters:
        image: a [N,M,3] tensor
        filename (str): name of the image 
    """
    image_path = os.path.join("../figures", image_name + ".jpg")
    print(image_path)

    imsave(image_path, image)

    

