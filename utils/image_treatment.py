from PIL import Image 
import numpy as np 
#charger une image 

def preprocess_image(image_path, new_size = (256, 256)):
    image = Image.open(image_path).convert('L')
    image = image.resize(new_size)
    image = image / np.sum(image)  # Normalisation pour que la somme des pixels = 1
    
    return image

def image_to_distribution(image, reduc=5):
    # Réduire la taille des images pour simplifier le traitement
    image_array = np.array(image)
    reduce_image = image_array[::reduc, ::reduc]

    reduce_image[reduce_image < np.mean(reduce_image)] = 0 #pour revenir a un vrai noir 

    # Obtenir la taille des images
    sz = reduce_image.shape[0]
    XX, YY = np.meshgrid(np.arange(sz), np.arange(sz))

    return np.stack((XX[reduce_image == 0], -YY[reduce_image == 0]), 1) * 1.0
