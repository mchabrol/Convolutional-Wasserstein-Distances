from utils.image_treatment import preprocess_image
from wasserstein.texture_mixing_new import compute_texture_mixing, compute_optimal_transport_barycenter, build_pyramid, compute_optimal_assignment, compute_optimal_transport, build_pyramid_barycenters, pyramid_projection
import matplotlib.pyplot as plt
import numpy as np
import pyrtools as pt
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks
from tqdm import tqdm



def build_blocks(pyramid, block_shape = (4, 4, 3)):
    blocks = {}
    for key in pyramid.keys():
        # Diviser l'image en blocs
        blocks[key] = view_as_blocks(pyramid[key], block_shape=block_shape)
    return(blocks)

def block_projection(block_wn, block_barycenter):
    """
    Compute the projections of white noise pyramid coefficients on barycenter pyramid coefficients (see 13 in paper)

    Parameters:
    - pyramid_wn (dict): white noise pyramid coefficients
    - pyramid_barycenter (dict): barycenter pyramid coefficients

    Returns:
    - pyramid_wn (dict): projection of white noise pyramid coefficients on barycenter pyramid
    """
    for key in block_wn.keys():
        size = block_wn[key].shape[0]
        projection = compute_optimal_assignment(block_wn[key].reshape(-1, 4*4*3), block_barycenter[key].reshape(-1, 4*4*3)).reshape(size, size, 4, 4, 3)
        block_wn[key] = projection
    
    return(block_wn)


def texture_mixing_high_dim(textures, rho, n_neighbor = 4, num_scales = 4, num_orientations = 4):
    size = textures[0].shape[0]
    noise = np.random.randn(size, size, 3)

    #calcul de Y
    Y = compute_optimal_transport_barycenter(noise.reshape(-1, 3), rho, [x.reshape(-1, 3) for x in textures], iterations=100).reshape(size, size, 3)

    #calculs de Y_l_j 
    pyramids = []
    for texture in textures:
        pyramids.append(build_pyramid(texture, num_scales=num_scales, num_orientations=num_orientations)) #returns a dico with pyramid for R,G,B
    #compute pyramid coefficients for white noise
    pyramid_wn = build_pyramid(noise, num_scales=num_scales, num_orientations=num_orientations)

    # construction des blocks C_l_j_N
    blocks = [build_blocks(x) for x in pyramids]
    block_wn = build_blocks(pyramid_wn)

    # calcul du barycentre C_l_N 
    block_barycenter = {}
    for key in tqdm(block_wn.keys()):
        size = block_wn[key].shape[0]
        block_barycenter[key] = compute_optimal_transport_barycenter(block_wn[key].reshape(-1,n_neighbor*n_neighbor*3), rho, [x[key].reshape(-1,n_neighbor*n_neighbor*3) for x in blocks])
        block_barycenter[key] = block_barycenter[key].reshape(size, size, 1, n_neighbor, n_neighbor, 3)

    # calcul du barycentre Y_l
    pyramid_barycenter = build_pyramid_barycenters(pyramid_wn, pyramids, rho, num_scales = num_scales, num_orientations = num_orientations)
    
    #(13), see article
    #noise = spectrum_constraint(noise, Y)
    pyramid_wn = build_pyramid(noise, num_scales=num_scales, num_orientations=num_orientations)
    c_lm = pyramid_projection(pyramid_wn, pyramid_barycenter)

    block_c_lm = build_blocks(c_lm)

    new_coeffs = block_projection(block_c_lm, block_barycenter)
    
    for key in new_coeffs.keys():
        size = new_coeffs[key].shape[0]
        new_coeffs[key] = new_coeffs[key].reshape(size*n_neighbor,size*n_neighbor,3)

    pyramid_barycenter_r = {}
    pyramid_barycenter_g = {}
    pyramid_barycenter_b = {}
    for key in pyramid_barycenter.keys(): #pour chaque coefficient (qui est pour l'instant en RGB)
        #il faut extraire les coefficients pour R, G et B
        pyramid_barycenter_r[key] = new_coeffs[key][:,:,0]
        pyramid_barycenter_g[key] = new_coeffs[key][:,:,1]
        pyramid_barycenter_b[key] = new_coeffs[key][:,:,2]

    #puis on reconstruit les images a partir de chaque pyramide R, G et B
    size = textures[0].shape[0]
    noise_for_pyr = np.random.randn(size, size)
    noisy_pyr = pt.pyramids.SteerablePyramidFreq(noise_for_pyr, height=4, order=4-1)

    noisy_pyr.pyr_coeffs = pyramid_barycenter_r
    reconstructed_pyr_r = noisy_pyr.recon_pyr()
    noisy_pyr.pyr_coeffs = pyramid_barycenter_g
    reconstructed_pyr_g = noisy_pyr.recon_pyr()
    noisy_pyr.pyr_coeffs = pyramid_barycenter_b
    reconstructed_pyr_b = noisy_pyr.recon_pyr()

    #f_tilde(k)
    reconstructed_pyr = np.stack((reconstructed_pyr_r, reconstructed_pyr_g, reconstructed_pyr_b), axis = -1)

    #f_k+1
    final_texture = compute_optimal_transport(reconstructed_pyr.reshape(-1,3), Y.reshape(-1, 3),iterations=50).reshape(size, size, 3)
    noise = final_texture

    return(final_texture.astype(int))



#exemple où ça marche moins bien
image_path3 = 'data/161.gif'
image_path4 = 'data/Wall.jpg'
size = 64
image3 = preprocess_image(image_path3, new_size = (size,size))
image4 = preprocess_image(image_path4, new_size = (size,size))
rho = [0.9, 0.1]
textures = [image3, image4]

synthesis_without_patch = compute_texture_mixing(textures, rho)
synthesis_with_patch = texture_mixing_high_dim(textures, rho, n_neighbor = 4, num_scales = 4, num_orientations = 4)

def plot(image, synth_without, synth_with):
    plt.figure(figsize=(6, 3))
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title('original')
    plt.subplot(1,3,2)
    plt.imshow(synth_without)
    plt.title('without')
    plt.subplot(1,3,3)
    plt.imshow(synth_with)
    plt.title('with')
    plt.show()
    #plt.close()

plot(image3, synthesis_without_patch, synthesis_with_patch)
