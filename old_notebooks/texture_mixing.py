import numpy as np
import pyrtools as pt
from .basic_wasserstein import compute_sliced_wass_barycenter
from joblib import Parallel, delayed

def compute_steerable_pyramid_coeffs(image, num_scales=3, num_orientations=4):
    """
    Compute steerable pyramid coefficients with specified orientations using pyrtools.
    
    Parameters:
    - image: 2D numpy array, input grayscale image.
    - num_scales: int, number of scales.
    - num_orientations: int, number of orientations.

    Returns:
    - coeffs: Dictionary of coefficients organized by scale and orientation.
    """
    # Initialize the steerable pyramid
    pyramid = pt.pyramids.SteerablePyramidFreq(image, height=num_scales, order=num_orientations-1)

    return pyramid.pyr_coeffs

def compute_3D_wavelets_coeffs(image, num_scales=4, num_orientations=4):
    """
    Compute wavelets coefficients (highpass, bandpass, low-residuals) for the 3 channels (R,G,B) of an image
    
    Parameters:
    - image: 2D numpy array, input grayscale image.
    - num_scales: int, number of scales.
    - num_orientations: int, number of orientations.

    Returns:
    - wavelets_coeffs: Dictionary of coefficients organized by channel (R,G,B) and then by bandpass (highpass, bandpass -scale and orientation- and low residual).
    """
    wavelets_coeffs = {}
    rgb = ['R','G','B']
    for channel in range(3):
        wavelets_coeffs[rgb[channel]] = compute_steerable_pyramid_coeffs(image[:, :, channel], num_scales=num_scales, num_orientations=num_orientations)
    return(wavelets_coeffs)

def compute_wavelet_coeffs_barycenter(textures, num_scales=4, num_orientations=4):
    """
    Compute the barycenter of wavelet coefficients for RGB channels.
    
    Parameters:
    - textures: 3D numpy array, input RGB image.
    - num_scales: int, number of scales.
    - num_orientations: int, number of orientations.
    
    Returns:
    - bar_wavelet_coeffs_RGB: Dictionary of barycenters of wavelet coefficients by channel (R, G, B) and then by highpass/bandpass/lowresidual.
    """
    RGB = ['R', 'G', 'B']
    bar_wavelet_coeffs_RGB = {rgb: {} for rgb in RGB}  # Initialize each channel's dictionary

    # Compute wavelet coefficients for all textures
    wavelets_coeffs = [compute_3D_wavelets_coeffs(image, num_scales, num_orientations) for image in textures]

    # Define a helper function to compute barycenter for each color channel and coefficient type
    def compute_barycenter(rgb, k):
        distributions = [w[rgb][k].reshape(-1, 1) for w in wavelets_coeffs]
        n = int(np.sqrt(distributions[0].shape[0]))
        barycenter = compute_sliced_wass_barycenter(distributions, rho=None).reshape(n, n)
        return rgb, k, barycenter

    # Use Parallel to compute barycenters for each RGB channel and coefficient type in parallel
    results = Parallel(n_jobs=-1)(
        delayed(compute_barycenter)(rgb, k)
        for rgb in RGB
        for k in wavelets_coeffs[0][rgb].keys()
    )

    # Populate the results in the dictionary
    for rgb, k, barycenter in results:
        bar_wavelet_coeffs_RGB[rgb][k] = barycenter

    return bar_wavelet_coeffs_RGB

def compute_textures_barycenter(textures):
    """
    Compute the barycenter of wavelet coefficients for RGB channels.
    
    Parameters:
    - textures: 3D numpy array, input RGB image.
    - num_scales: int, number of scales.
    - num_orientations: int, number of orientations.
    
    Returns:
    - bar_wavelet_coeffs_RGB: Dictionary of barycenters of wavelet coefficients by channel (R, G, B) and then by highpass/bandpass/lowresidual.
    """
    RGB = ['R', 'G', 'B']
    bar_wavelet_coeffs_RGB = {rgb: {} for rgb in RGB}  # Initialize each channel's dictionary

    # Compute wavelet coefficients for all textures

    # Define a helper function to compute barycenter for each color channel and coefficient type
    def compute_barycenter_texture(rgb):
        # Iterate over all distributions associated with the specified 'rgb' key
        distributions = [w[rgb] for w in textures]
        n = int(np.sqrt(distributions[0].shape))
        
        # Compute the barycenter of the distributions
        barycenter = compute_sliced_wass_barycenter(distributions, rho=None).reshape(n, n)
        
        return rgb, barycenter

    # Use Parallel to compute barycenters for each RGB channel and coefficient type in parallel
    results = Parallel(n_jobs=-1)(
        delayed(compute_barycenter_texture)(rgb)
        for rgb in RGB)

    # Populate the results in the dictionary
    for rgb, barycenter in results:
        bar_wavelet_coeffs_RGB[rgb] = barycenter

    return bar_wavelet_coeffs_RGB








