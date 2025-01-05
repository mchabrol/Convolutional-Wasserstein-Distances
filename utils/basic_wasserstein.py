import numpy as np
import ot
import torch

def compute_wasserstein_distance(x1, x2): 
    # Set uniform weights for each distribution
    w1 = np.ones(len(x1)) / len(x1)
    w2 = np.ones(len(x2)) / len(x2)

    # Compute the cost matrix (squared Euclidean distance between each pair of points)
    M = ot.dist(x1, x2, metric='sqeuclidean')

    # Compute the Wasserstein distance using ot.emd2
    return np.sqrt(ot.emd2(w1, w2, M))

def compute_wasserstein_barycenter(distributions, weights = None, k = 200, X_init = None):
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_torch = [torch.tensor(x).to(device=device) for x in distributions]

    if rho is None: 
        n = len(distributions)
        rho = n*[1/n]
    
    if xbinit is None:
        #xbinit = np.random.randn(500, 2) * 10 + 16 #initialization
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
    return(xb)

def sliced_projection(X0, Y):
    """
    X0 : image a projeter 
    Y : ce sur quoi on veut projeter 
    """
    Y_distrib = [Y]
    proj = compute_sliced_wass_barycenter(Y_distrib, rho = None, lr = 1e3, k = 200, d = 2, nb_iter_max = 50, xbinit = X0)
    return(proj)