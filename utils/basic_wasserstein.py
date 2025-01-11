### Utils functions for computing Wasserstein barycenters and distances


import numpy as np
import ot
import torch

def compute_wasserstein_distance(x1, x2): 
    """
    Args:
        x1 (ndarray): An array representing the first distribution, where each element corresponds to a sample point.
        x2 (ndarray): An array representing the second distribution, where each element corresponds to a sample point.

    Returns:
        float: The computed Wasserstein distance between the two distributions.
    """
    # Set uniform weights for each distribution
    w1 = np.ones(len(x1)) / len(x1)
    w2 = np.ones(len(x2)) / len(x2)

    # Compute the cost matrix (squared Euclidean distance between each pair of points)
    M = ot.dist(x1, x2, metric='sqeuclidean')

    # Compute the Wasserstein distance using ot.emd2
    return np.sqrt(ot.emd2(w1, w2, M))


def compute_wasserstein_barycenter(distributions, weights = None, k = 200, X_init = None):
    """
    Computes the Wasserstein barycenter of a set of distributions.

    Args:
        distributions (list of ndarray): A list of distributions, where each distribution is represented as 
            an array of sample points.
        weights (list of float, optional): A list of weights associated with each distribution. If `None`, 
            all distributions are assigned equal weights. Defaults to `None`.
        k (int, optional): The number of points to use for the barycenter's support. Defaults to `200`.
        X_init (ndarray, optional): Initial positions of the barycenter's support points. If `None`, 
            points are initialized randomly using a standard normal distribution. Defaults to `None`.

    Returns:
        ndarray: The computed Wasserstein barycenter represented as an array of shape `(k, n_features)`, 
        where `k` is the number of support points and `n_features` is the dimensionality of the distributions.
    """
    d = distributions[0].shape[1]
    if X_init is None:
        X_init = np.random.normal(0., 1., (k, d))

    if weights is None:
        weights = [1 / len(distributions)] * len(distributions)

    measures_weights = [ot.unif(dist.shape[0]) for dist in distributions]

    b = np.ones((k,)) / k  # weights of the barycenter (it will not be optimized, only the locations are optimized)

    X = ot.lp.free_support_barycenter(distributions, measures_weights, X_init, b, weights = weights)
    return(X)

def compute_wasserstein_sliced_distance(x1, x2, n_seed=20, n_projections_arr = np.logspace(0, 3, 10, dtype=int)):
    """
    Computes the Sliced Wasserstein Distance (SWD) between two distributions.

    Args:
        x1 (ndarray): An array representing the first distribution, where each element is a sample point.
        x2 (ndarray): An array representing the second distribution, where each element is a sample point.
        n_seed (int, optional): The number of random seeds to use for the projections. 
            This determines the variability in the distance computation. Defaults to `20`.
        n_projections_arr (ndarray, optional): An array of integers specifying the number of random projections 
            to evaluate at each step. Defaults to `np.logspace(0, 3, 10, dtype=int)`.

    Returns:
        float: The mean Sliced Wasserstein Distance (SWD) averaged over all seeds and projection counts.
    """
    res = np.empty((n_seed, len(n_projections_arr)))

    # Set uniform weights for each distribution
    w1 = np.ones(len(x1)) / len(x1)
    w2 = np.ones(len(x2)) / len(x2)


    # Compute Sliced Wasserstein Distance for each seed and each number of projections
    for seed in range(n_seed):
        for i, n_projections in enumerate(n_projections_arr):
            # Compute SWD with specific number of projections and seed
            res[seed, i] = ot.sliced_wasserstein_distance(x1, x2, w1, w2, n_projections, seed=seed)

    # Calculate the mean SWD over all seeds and projection counts
    return np.mean(np.mean(res, axis=0))


def compute_sliced_wass_barycenter(distributions, rho = None, lr = 1e3, k = 200, nb_iter_max = 50, xbinit = None):
    """
    Computes the Sliced Wasserstein Barycenter of given distributions using gradient descent.

    Args:
        distributions (list of ndarray): List of input distributions, each of shape `(n_samples, n_features)`.
        rho (list of float, optional): Weights for each distribution. Defaults to equal weights.
        lr (float, optional): Learning rate for gradient descent. Defaults to `1e3`.
        k (int, optional): Number of support points in the barycenter. Defaults to `200`.
        nb_iter_max (int, optional): Maximum number of iterations. Defaults to `50`.
        xbinit (ndarray, optional): Initial support points for the barycenter. Randomly initialized if `None`.

    Returns:
        ndarray: Computed barycenter, shape `(k, n_features)`.

    Notes:
        - Utilizes PyTorch for GPU acceleration and gradient computation.
        - Random projections in sliced Wasserstein are seeded for reproducibility.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_torch = [torch.tensor(x).to(device=device) for x in distributions]

    if rho is None: 
        n = len(distributions)
        rho = n*[1/n]
    
    if xbinit is None:
        xbinit = np.random.normal(0., 1., distributions[0].shape)
    xbary_torch = torch.tensor(xbinit).to(device=device).requires_grad_(True)


    x_all = np.zeros((nb_iter_max, xbary_torch.shape[0], xbary_torch.shape[1]))

    loss_iter = []

    # generator for random permutations
    gen = torch.Generator(device=device)
    gen.manual_seed(42)


    for i in range(nb_iter_max):

        loss = 0
        for i, x in enumerate(x_torch):
            loss += rho[i] * ot.sliced_wasserstein_distance(xbary_torch, x, n_projections=50, seed=gen)
        loss_iter.append(loss.clone().detach().cpu().numpy())
        loss.backward()

        # performs a step of projected gradient descent
        with torch.no_grad():
            grad = xbary_torch.grad
            xbary_torch -= grad * lr  # / (1 + i / 5e1)  # step
            xbary_torch.grad.zero_()
            x_all[i, :, :] = xbary_torch.clone().detach().cpu().numpy()

    xb = xbary_torch.clone().detach().cpu().numpy()
    return(xb, x_all)

def sliced_projection(X0, Y):
    """
    Projects an image onto a target distribution using the Sliced Wasserstein Barycenter.

    Args:
        X0 (ndarray): Initial image to be projected, shape `(n_samples, n_features)`.
        Y (ndarray): Target distribution, shape `(m_samples, n_features)`.

    Returns:
        ndarray: Projected image, shape `(n_samples, n_features)`.
    """
    Y_distrib = [Y]
    proj = compute_sliced_wass_barycenter(Y_distrib, rho = None, lr = 1e3, k = 200, d = 2, nb_iter_max = 50, xbinit = X0)
    return(proj)