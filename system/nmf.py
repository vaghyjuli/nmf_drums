import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

EPS = 2.0 ** -52

## based on https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S3_NMFbasic.html
def NMF(V, W_init, params, L = 1000, threshold = 0.001):
    """
        Non-Negative Matrix Factorization.

        Args:
            V (np.ndarray) : A 2D numpy array of size K x N, representing the magnitude
                spectrogram with K spectral bands on N time steps.
            W_init (np.ndarray) : A 3D numpy array of size K x R x T, representing template
                magnitude spectrograms with K spectral bands on T time steps, for each
                of the R instruments.
            params (dict) : Dictionary of parameters, defined in main.py.
            L (int) : The number of NMFD iterations.
            threshold (float) : If the element-wise difference between W and W' and between
                H and H' is < threshold, the gradient descent stops. 

        Returns:
            V_approx (np.ndarray) : A 2D numpy array of size K x N,  representing the
                magnitude spectrogram approximated by the NMFD components.
            W (np.ndarray) : A 3D numpy array of size K x R x T, representing the
                (adapted) template magnitude spectrograms.
            H (np.ndarray) : A 2D numpy array of size R x N, representing the activations
                for each of the R instruments over N time steps.
    """
    K, N = V.shape
    K, R = W_init.shape

    # add Q uniformly initialized additional noise template components to W
    W_init = np.append(W_init, np.ones((K, params["addedCompW"])), axis=1)
    # W_init = np.append(W_init, np.random.rand(K, params["addedCompW"]) + EPS, axis=1)
    R += params["addedCompW"]

    if ("initH" not in params) or (params["initH"] == "uniform"):
        H = np.ones((R, N))
    elif params["initH"] == "random":
        H = np.random.rand(R, N)

    W = deepcopy(W_init)
    onesMatrix = np.ones((K, N))

    for iteration in range(L):
        V_approx = W.dot(H)
        Q = V / (V_approx + EPS)

        ## Equations 2.3 and 2.4 ##
        H_prev = deepcopy(H)
        H = H * (W.transpose().dot(Q) / (W.transpose().dot(onesMatrix) + EPS))
        W_prev = deepcopy(W)
        W = W * (Q.dot(H.transpose()) / (onesMatrix.dot(H.transpose()) + EPS))

        ## Equation 2.7 ##
        if params["fixW"] == "fixed":
            W[:, :R-params["addedCompW"]] = W_init[:, :R-params["addedCompW"]]

        ## Equation 2.5 ##
        elif params["fixW"] == "semi":
            alpha = (iteration / L)**params["beta"]
            W[:, :R-params["addedCompW"]] = (1-alpha) * W_init[:, :R-params["addedCompW"]] + alpha * W[:, :R-params["addedCompW"]]

        W_diff = np.linalg.norm(W - W_prev, ord=2)
        H_diff = np.linalg.norm(H - H_prev, ord=2)
        if H_diff < threshold and W_diff < threshold:
            break

    V_approx = W.dot(H)
    return V_approx, W, H