import numpy as np
import torch
from kymatio.numpy import Scattering2D
import pyrtools as pt

def compute_steerable_wavelet_atoms_with_extra_frames(image, J=4, L=4):
    """
    Compute steerable wavelet atoms ψ_{l,n} using Kymatio's Scattering2D transform,
    including low-pass residual and high-frequency details.

    Parameters:
    - image: 2D numpy array, input grayscale image.
    - J: int, the number of dyadic scales (2^J is the largest scale).
    - L: int, number of orientations (angular resolution).

    Returns:
    - wavelet_coeffs: list of dictionaries with 'scale', 'orientation', 'coefficients'
    """
    # Initialize Scattering Transform
    scattering = Scattering2D(J=J, shape=image.shape, L=L)
    
    # Convert image to tensor
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Perform scattering transform to get wavelet coefficients
    S = scattering(image_tensor)
    
    # Convert back to numpy and organize by scale and orientation
    S = S.squeeze().detach().numpy()
    
    # Retrieve metadata to identify low-pass residual
    meta = scattering.meta()
    
    wavelet_coeffs = []
    
    # Step 1: Extract main scale-orientation coefficients
    for i, (order, scale, orientation) in enumerate(zip(meta['order'], meta['scale'], meta['theta'])):
        # Skip the low-pass component initially
        if order == 0:
            continue
        coeffs = S[i]  # Specific coefficients for scale and orientation
        wavelet_coeffs.append({
            'scale': scale,
            'orientation': orientation,
            'coefficients': coeffs
        })
    
    # Step 2: Extract low-pass residual (coarse scale frame)
    # According to meta['order'], the low-pass residual is the first item (order == 0)
    low_pass_residual = S[0]  # Low-pass residual based on metadata
    wavelet_coeffs.append({
        'scale': 'low-pass',
        'orientation': None,
        'coefficients': low_pass_residual
    })
    
    # Step 3: Extract high-frequency details (difference with low-pass approximation)
    # Ensure low_pass_residual is resized to the original image shape if needed
    if low_pass_residual.shape != image.shape:
        low_pass_residual_resized = np.resize(low_pass_residual, image.shape)
    else:
        low_pass_residual_resized = low_pass_residual
    
    high_frequency_details = image - low_pass_residual_resized
    wavelet_coeffs.append({
        'scale': 'high-pass',
        'orientation': None,
        'coefficients': high_frequency_details
    })
    
    return wavelet_coeffs


def initialize_random_image(size=(256, 256), channels=3):
    """Initialize a random white noise image."""
    return np.random.rand(*size, channels)



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
    
    # Print keys to confirm structure (for debugging)
    #print("Available keys in pyramid coefficients:", pyramid.pyr_coeffs.keys())

    # Extract coefficients and organize them by scale and orientation
    coeffs = {
        'highpass': pyramid.pyr_coeffs['residual_highpass'],  # Adjust based on key inspection
        'bandpass': {},  # Dictionary to hold bandpass coefficients by scale and orientation
        'lowpass': pyramid.pyr_coeffs['residual_lowpass']    # Adjust based on key inspection
    }
    
    for scale in range(num_scales):
        coeffs['bandpass'][scale] = {}
        for orientation in range(num_orientations):
            # Access bandpass coefficients at each scale and orientation
            coeffs['bandpass'][scale][orientation] = pyramid.pyr_coeffs[(scale, orientation)]
    
    return coeffs

