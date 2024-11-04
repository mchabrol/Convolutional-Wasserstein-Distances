import numpy as np
import torch
from kymatio.numpy import Scattering2D
import pyrtools as pt
import tqdm
from .basic_wasserstein import compute_wasserstein_sliced_distance, compute_sliced_wass_barycenter
from joblib import Parallel, delayed

def initialize_random_image(size=(256, 256), channels=3):
    """Initialize a random white noise image f^(0)."""
    return np.random.rand(*size, channels)

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
    image_r, image_g, image_b = image.split()

    image_r = np.array(image_r)
    image_g = np.array(image_g)
    image_b = np.array(image_b)

    pyr_r = pt.pyramids.SteerablePyramidFreq(image_r, height=num_scales, order=num_orientations-1)
    pyr_g = pt.pyramids.SteerablePyramidFreq(image_g, height=num_scales, order=num_orientations-1)
    pyr_b = pt.pyramids.SteerablePyramidFreq(image_b, height=num_scales, order=num_orientations-1)

    return {'R': pyr_r.pyr_coeffs, 'G': pyr_g.pyr_coeffs, 'B': pyr_b.pyr_coeffs}

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
    wavelets_coeffs = [compute_3D_wavelets_coeffs(image) for image in textures]

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

def compute_sliced_wass_barycenter_pixels(textures, weights=None):
    # Initialisation de la texture barycentre
    barycenter = {}
    
    # Noms des canaux RGB
    RGB = ['R', 'G', 'B']
    
    for i, rgb in tqdm(enumerate(RGB)):
        # Extraire le canal correspondant et le reshaper pour avoir un vecteur colonne
        channel_textures = [
            textures[0][:, :, i].astype(np.float64).reshape(-1, 1),  
            textures[1][:, :, i].astype(np.float64).reshape(-1, 1)  
        ]
        print(f'Length of channel {rgb}: {len(channel_textures[0])}')

        # Calculer le barycentre
        n = int(np.sqrt(channel_textures[0].shape[0]))
        barycenter_result = compute_sliced_wass_barycenter(channel_textures, rho=weights).reshape(n, n)
        barycenter[rgb] = barycenter_result
    
    return barycenter

def generate_random_directions(dim, num_directions):
    """
    Generate a set of random unit directions in a given dimensional space.
    
    Parameters:
    - dim: int, dimension of the space.
    - num_directions: int, number of random directions to generate.
    
    Returns:
    - Array of shape (num_directions, dim), where each row is a unit vector.
    """
    directions = np.random.randn(num_directions, dim)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions

def compute_projected_wasserstein(X, Y, directions):
    """
    Compute the average Wasserstein distance between two point clouds
    along specified directions.
    
    Parameters:
    - X, Y: numpy arrays of shape (n_samples, dim), the two datasets to compare.
    - directions: numpy array of shape (num_directions, dim), the directions for projection.
    
    Returns:
    - Average Wasserstein distance over all directions.
    """
    num_directions = directions.shape[0]
    wasserstein_distances = []
    
    for i in range(num_directions):
        # Project both X and Y onto the current direction
        proj_X = X @ directions[i]
        proj_Y = Y @ directions[i]
        
        # Sort projections for 1D Wasserstein distance
        proj_X_sorted = np.sort(proj_X)
        proj_Y_sorted = np.sort(proj_Y)
        
        # Compute 1D Wasserstein distance
        distance = compute_wasserstein_sliced_distance(proj_X_sorted, proj_Y_sorted)
        wasserstein_distances.append(distance)
    
    # Return the average Wasserstein distance over all directions
    return np.mean(wasserstein_distances)

def approximate_projection(X, Y, num_directions=50, learning_rate=0.01, num_iterations=100):
    """
    Approximate projection of X onto the distribution Y using SGD to minimize
    the Wasserstein distance.
    
    Parameters:
    - X: numpy array of shape (n_samples, dim), the data to project.
    - Y: numpy array of shape (n_samples, dim), the target distribution.
    - num_directions: int, the number of random directions for slicing.
    - learning_rate: float, the SGD step size.
    - num_iterations: int, the number of SGD iterations.
    
    Returns:
    - X_proj: numpy array, the approximate projection of X onto Y.
    """
    # Make a copy of X to avoid modifying the original data
    X_proj = X.copy()
    dim = X.shape[1]
    
    for iteration in range(num_iterations):
        # Generate random directions
        directions = generate_random_directions(dim, num_directions)
        
        # Compute gradient: here, we approximate by measuring the Wasserstein distance
        loss = compute_projected_wasserstein(X_proj, Y, directions)
        
        # Update X_proj using a gradient approximation with respect to Wasserstein distance
        for i in range(num_directions):
            # Project X and Y onto the current direction
            proj_X = X_proj @ directions[i]
            proj_Y = Y @ directions[i]
            
            # Compute the difference in projections
            diff = np.mean(proj_X - proj_Y)
            
            # Adjust each point in X_proj along the current direction
            X_proj -= learning_rate * diff * directions[i]
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Wasserstein loss: {loss}")
    
    return X_proj


