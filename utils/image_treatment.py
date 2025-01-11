from PIL import Image 
import numpy as np 
import pywt 

def preprocess_image(image_path, new_size = (256, 256), end = "RGB"):
    """
    Preprocesses an image by resizing, converting color mode, and returning it as a NumPy array.

    Args:
        image_path (str): Path to the image file.
        new_size (tuple, optional): Target size for resizing as `(width, height)`. Defaults to `(256, 256)`.
        end (str, optional): Color mode for conversion (e.g., "RGB", "L"). Defaults to "RGB". L if we want black and white/grey

    Returns:
        ndarray: Preprocessed image as a NumPy array.
    """
    image = Image.open(image_path)
    image = image.convert(end)
    image = image.resize(new_size)
    image_array = np.array(image)
    return image_array

def image_to_distribution(image_array, reduc=5):
    """
    Converts an image array into a distribution by downsampling and thresholding.

    Args:
        image_array (ndarray): Input image as a NumPy array.
        reduc (int, optional): Factor by which to downsample the image. Defaults to 5.

    Returns:
        ndarray: Distribution of the image, with coordinates for pixels that are black (thresholded).
    """
    # Resize the image to simplify processing
    reduce_image = image_array[::reduc, ::reduc]
    reduce_image[reduce_image < np.mean(reduce_image)] = 0 # to have black
    # Get the images size 
    sz = reduce_image.shape[0]
    XX, YY = np.meshgrid(np.arange(sz), np.arange(sz))

    return np.stack((XX[reduce_image == 0], -YY[reduce_image == 0]), 1) * 1.0