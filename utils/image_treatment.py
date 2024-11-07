from PIL import Image 
import numpy as np 
import pywt 

def preprocess_image(image_path, new_size = (256, 256), end = "RGB"):
    image = Image.open(image_path)
    image = image.convert(end)
    image = image.resize(new_size)
    #image = image / np.sum(image)  # Normalisation pour que la somme des pixels = 1
    image_array = np.array(image)
    return image_array

def image_to_distribution(image_array, reduc=5):
    # Réduire la taille des images pour simplifier le traitement
    
    reduce_image = image_array[::reduc, ::reduc]

    reduce_image[reduce_image < np.mean(reduce_image)] = 0 #pour revenir a un vrai noir 

    # Obtenir la taille des images
    sz = reduce_image.shape[0]
    XX, YY = np.meshgrid(np.arange(sz), np.arange(sz))

    return np.stack((XX[reduce_image == 0], -YY[reduce_image == 0]), 1) * 1.0

""" def extract_wavelet_features(image, wavelet_name='db1', levels=3):
   
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet=wavelet_name, level=levels)
    
    # Gather wavelet coefficients as features
    features = []
    for level in range(1, levels + 1):
        # Approximation (low-pass)
        cA = coeffs[0] if level == 1 else None
        
        # Detail coefficients (horizontal, vertical, diagonal)
        cH, cV, cD = coeffs[level]
        
        # Optional: take magnitude or energy of coefficients for each component
        if cA is not None:
            features.append(np.abs(cA))  # Low-pass
        features.append(np.abs(cH))     # Horizontal details
        features.append(np.abs(cV))     # Vertical details
        features.append(np.abs(cD))     # Diagonal details

    # Flatten or aggregate features as needed (e.g., reshape for simpler handling)
    return features """