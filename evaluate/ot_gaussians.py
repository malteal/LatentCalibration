"evaluate ot ftag calibration"
import pyrootutils


root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import numpy as np
import logging
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
import torch as T

# from framework
from run.predict_ot import PredictionHandler

# internal
from tools.tools.hydra_utils import hydra_init
from scipy.linalg import sqrtm

import matplotlib
font = {'size'   : 20}

matplotlib.rc('font', **font)

log = logging.getLogger(__name__)


def optimal_transport_gaussian(mu1, Sigma1, mu2, Sigma2, x):
    """
    Computes the optimal transport map between two Gaussian distributions.

    Args:
        mu1 (numpy.ndarray): Mean of the first Gaussian distribution (d-dimensional vector).
        Sigma1 (numpy.ndarray): Covariance matrix of the first Gaussian distribution (dxd matrix).
        mu2 (numpy.ndarray): Mean of the second Gaussian distribution (d-dimensional vector).
        Sigma2 (numpy.ndarray): Covariance matrix of the second Gaussian distribution (dxd matrix).
        x (numpy.ndarray): Points to transport (dxN matrix, where N is the number of points).

    Returns:
        numpy.ndarray: Transported points (dxN matrix).
    """

    Sigma1_sqrt = sqrtm(Sigma1)
    
    A = np.linalg.inv(Sigma1_sqrt) @ sqrtm(Sigma1_sqrt @ Sigma2 @ Sigma1_sqrt) @ np.linalg.inv(Sigma1_sqrt)
    
    return mu2 + (x - mu1) @ A.T

if __name__ == '__main__':
    # %matplotlib widget
    config = hydra_init(str(root/"configs/eval_ot.yaml"), verbose=True)

    # n_bins=10
    selection = config['type_to_eval']
    gen_name = config["evaluation_sample"]

    # get paths to model
    path_to_eval = config['paths']
    nominal_path = path_to_eval[selection]['gauss']

    ######### get nominal prediction #########
    transport_handler = PredictionHandler(nominal_path, gen_name,
                                          config["bkg_scaler_str"],
                                          force_prediction = False,
                                          allow_prediction = False)

    data = transport_handler.load_prediction()
    
    Sigma2 = np.load(f'{nominal_path}/cov.npy')
    mu2 = np.load(f'{nominal_path}/mean.npy')
    
    Sigma1 = np.eye(16)
    mu1 = np.zeros(16)
    
    source = data['source'].values[:, :16]
    transport = data['eval_transport'].values[:, :16]

    ot_source = optimal_transport_gaussian(mu1, Sigma1, mu2, Sigma2, source)
    
    # calcuate the distance between ana-ot and ml-ot
    print('Source:', np.linalg.norm(ot_source-source, axis=1).mean(0))
    print('Transport:', np.linalg.norm(ot_source-transport, axis=1).mean(0))
    print('Length of diagonal:', np.sqrt(np.sum(np.diag(Sigma2)**2)))

    # # calculate kl div
    # mu2 = T.tensor(mu2)
    # Sigma2 = T.tensor(Sigma2)
    
    # from torch.distributions.multivariate_normal import MultivariateNormal
    # # Create a multivariate normal distribution
    # multivariate_gaussian = MultivariateNormal(mu2, Sigma2)

    # # Sample from the distribution
    # sample = multivariate_gaussian.sample()
    # prob = multivariate_gaussian.log_prob(T.tensor(ot_source)).exp()

    # (prob * (multivariate_gaussian.log_prob(T.tensor(ot_source))- multivariate_gaussian.log_prob(T.tensor(transport)))).mean()
    
    # multivariate_gaussian.log_prob(T.tensor(transport)).mean()

    # (prob * (multivariate_gaussian.log_prob(T.tensor(ot_source))- multivariate_gaussian.log_prob(T.tensor(source)))).mean()
    # multivariate_gaussian.log_prob(T.tensor(transport)).mean()
        
    
    
    
    