import numpy as np
import torch
from kymatio.numpy import Scattering2D

def compute_steerable_wavelet_atoms(image, J=4, L=4):
    """
    Compute steerable wavelet atoms ψ_{l,n} using Kymatio's Scattering2D transform.
    
    Parameters:
    - image: 2D numpy array, input grayscale image.
    - J: int, the maximum dyadic scale (2^J is the largest scale).
    - L: int, number of orientations (angular resolution).
    
    Returns:
    - wavelet_coeffs: list of dictionaries with 'scale', 'orientation', 'position', and 'coefficients'
    """
    # Initialize Scattering Transform
    scattering = Scattering2D(J=J, shape=image.shape, L=L)
    
    # Convert image to tensor and normalize
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    # Compute scattering transform (wavelet decomposition)
    S = scattering(image_tensor)
    
    # Convert result back to numpy
    S = S.squeeze().detach().numpy()
    
    # Organize coefficients by scale, orientation, and position
    wavelet_coeffs = []
    for j in range(J):  # Scales
        for l in range(L):  # Orientations
            coeffs = S[j * L + l]  # Extract coefficients for specific scale and orientation
            wavelet_coeffs.append({
                'scale': j,
                'orientation': l * (np.pi / L),  # Convert index to angle in radians
                'coefficients': coeffs
            })
    return wavelet_coeffs

def initialize_random_image(size=(256, 256), channels=3):
    """Initialize a random white noise image."""
    return np.random.rand(*size, channels)