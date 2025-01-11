from utils.basic_wasserstein import compute_sliced_wass_barycenter, compute_wasserstein_barycenter
import matplotlib.pyplot as plt
import imageio

def sliced_projection(X0, Y, k=200):
    """
    Projects an image (X0) onto a target distribution (Y) using the Sliced Wasserstein barycenter method.

    Args:
        X0 (ndarray): The image to be projected.
        Y (ndarray): The target distribution onto which the image will be projected.
        k (int, optional): The number of points for the barycenter calculation. Defaults to 200.

    Returns:
        ndarray: The projected image.
    """
    Y_distrib = [Y]
    proj,_ = compute_sliced_wass_barycenter(Y_distrib, rho = None, lr = 1e3, k = k, nb_iter_max = 50, xbinit = X0)
    return(proj)


def simple_projection(X0, Y, k=200):
    """
    Projects an image (X0) onto a target distribution (Y) using the Wasserstein barycenter method.

    Args:
        X0 (ndarray): The image to be projected.
        Y (ndarray): The target distribution onto which the image will be projected.
        k (int, optional): The number of points for the barycenter calculation. Defaults to 200.

    Returns:
        ndarray: The projected image.
    """
    Y_distrib = [Y]
    proj = compute_wasserstein_barycenter(Y_distrib, weights= None, k = k, X_init = X0)
    return(proj)

def resize_array(arr):
    range_col1 = (0, 50)
    range_col2 = (-50, 0)

    col1_min, col1_max = arr[:, 0].min(), arr[:, 0].max()
    arr[:, 0] = range_col1[0] + (arr[:, 0] - col1_min) * (range_col1[1] - range_col1[0]) / (col1_max - col1_min)

    col2_min, col2_max = arr[:, 1].min(), arr[:, 1].max()
    arr[:, 1] = range_col2[0] + (arr[:, 1] - col2_min) * (range_col2[1] - range_col2[0]) / (col2_max - col2_min)

    return arr

def create_gif(x_all, y_points, filename="outputs/projection.gif"):
    """
    Creates a GIF showing the evolution of X towards Y.
    
    Parameters:
        x_all : ndarray
            Array containing the evolution of X at each iteration.
        y_points : ndarray
            Points of Y for reference.
        filename : str
            Name of the generated GIF file.
    """
    images = []
    for i, x in enumerate(x_all):
        plt.figure(figsize=(6, 6))
        
        ax = plt.gca()
        ax.set_facecolor('lightgray')
        
        plt.scatter(x[:, 0], x[:, 1], c='blue', label='Projected X', alpha=0.6)
        plt.scatter(y_points[:, 0], y_points[:, 1], c='red', label='Target Y', alpha=0.2)
        plt.legend(loc='upper right')
        plt.xlim(0, 50)
        plt.ylim(-50, 0)
        
        plt.axis('off')
        
        plt.savefig("frame.png", bbox_inches='tight', pad_inches=0)
        images.append(imageio.imread("frame.png"))
        plt.close()
    
    imageio.mimsave(filename, images, fps=20, loop=5)


def resize_array(arr):
    """Preprocessing and resizing the array"""
    range_col1 = (0, 50)
    range_col2 = (-50, 0)

    col1_min, col1_max = arr[:, 0].min(), arr[:, 0].max()
    arr[:, 0] = range_col1[0] + (arr[:, 0] - col1_min) * (range_col1[1] - range_col1[0]) / (col1_max - col1_min)

    col2_min, col2_max = arr[:, 1].min(), arr[:, 1].max()
    arr[:, 1] = range_col2[0] + (arr[:, 1] - col2_min) * (range_col2[1] - range_col2[0]) / (col2_max - col2_min)

    return arr

