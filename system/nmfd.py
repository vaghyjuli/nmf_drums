import numpy as np
from copy import deepcopy

EPS = 2.0 ** -52

## based on https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
def NMFD(V, P_init, params, L=50, threshold = 0.001):
    """
        Non-Negative Matrix Factor Deconvolution.

        Parameters:
            V (np.ndarray) : A 2D numpy array of size K x N, representing the magnitude
                spectrogram with K spectral bands on N time steps.
            P_init (np.ndarray) : A 3D numpy array of size K x R x T, representing template
                magnitude spectrograms with K spectral bands on T time steps, for each
                of the R instruments.
            params (dict) : Dictionary of parameters, defined in main.py.
            L (int) : The number of NMFD iterations.
            threshold (float) : If the element-wise difference between P and P' and between
                H and H' is < threshold, the gradient descent stops.

        Returns:
            V_approx (np.ndarray) : A 2D numpy array of size K x N,  representing the
                magnitude spectrogram approximated by the NMFD components.
            P (np.ndarray) : A 3D numpy array of size K x R x T, representing the
                (adapted) template magnitude spectrograms.
            H (np.ndarray) : A 2D numpy array of size R x N, representing the activations
                for each of the R instruments over N time steps.
    """

    # num of spectral bands, num of NMFD components, num of time frames in the component templates
    K, R, T = P_init.shape
    # num of spectral bands, num of time frames in the full spectrogram
    K, N = V.shape

    #P_init = np.append(P_init, np.random.rand(K, params["addedCompW"], T) + EPS, axis=1)
    P_init = np.append(P_init, np.ones((K, params["addedCompW"], T)), axis=1)
    R += params["addedCompW"]

    # initalize the activation matrix
    if ("initH" not in params) or params["initH"] == "uniform":
        H = np.ones((R, N))
    elif params["initH"] == "random":
        H = np.random.rand(R, N)

    if params["beta"] == None:
        params["beta"] = 4
        print("beta 4")
    
    P = deepcopy(P_init)

    # helper matrix of all ones, denoted as J
    onesMatrix = np.ones((K, N))

    for iteration in range(L):
        H_prev = deepcopy(H)
        P_prev = deepcopy(P)
        V_approx = convModel(P, H)

        # compute the ratio of the input to the model
        Q = V / (V_approx + EPS)

        # accumulate activation updates here
        multH = np.zeros((R, N))

        ## Equations 2.8 and 2.9 ##
        for t in range(T):
            # use tau for shifting and t for indexing
            tau = deepcopy(t)

            transpH = shiftOperator(H, tau).T

            multP = Q @ transpH / (onesMatrix @ transpH + EPS)
            P[:, :, t] *= multP

            if params["fixW"] == "fixed":
                P[:, :R-params["addedCompW"], t] = P_init[:, :R-params["addedCompW"], t]

            ## Equation 2.5 ##            
            elif params["fixW"] == "semi":
                alpha = (iteration / L)**params["beta"]
                P[:, :R-params["addedCompW"], t] = (1-alpha) * P_init[:, :R-params["addedCompW"], t] + alpha * P[:, :R-params["addedCompW"], t]

            transpP = P[:, :, t].T
            addP = (transpP @ shiftOperator(Q, -tau)) / (transpP @ onesMatrix + EPS)
            multH += addP
        H *= multH

        H_diff = np.linalg.norm(np.abs(H - H_prev), ord=2)
        P_diff = np.linalg.norm(np.mean(np.abs(P - P_prev), axis=2), ord=2)
        if H_diff < threshold and P_diff < threshold:
            break

    V_approx = convModel(P, H)
    return V_approx, P, H

## taken from https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
def convModel(P, H):
    """
        Calculate convolutive approximation of the original magnitude spectrogram V,
        see Equation 1.12.

        Args:
            P (np.ndarray) : A 3D numpy array of size K x R x T, representing template
                magnitude spectrograms with K spectral bands on T time steps, for each
                of the R instruments.
            H (np.ndarray) : A 2D numpy array of size R x N, representing the activations
                for each of the R instruments over N time steps.

        Returns:
            V_approx (np.ndarray) : A 2D numpy array of size K x N, representing the
                magnitude spectrogram approximated by the NMFD components.
    """
    K, R, T = P.shape
    R, N = H.shape

    V_approx = np.zeros((K, N))

    for k in range(T):
        multResult = P[:, :, k] @ shiftOperator(H, k)
        V_approx += multResult

    V_approx += EPS

    return V_approx

## taken from https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
def shiftOperator(A, shiftAmount):
    """
        Shift operator as defined by Smaragdis (2004), see equation 1.13.

        Args:
            A (np.ndarray) : The matrix to be shifted
            shiftAmount (int) : "Positive numbers shift to the right, negative numbers
                shift to the left, zero leaves the matrix unchanged" (https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/)

        Returns:
            shifted (np.ndarray) : The shifted matrix
    """

    numRows, numCols = A.shape
    shiftAmount = np.sign(shiftAmount) * min(abs(shiftAmount), numCols)
    shifted = np.roll(A, shiftAmount, axis=-1)

    if shiftAmount < 0:
        shifted[:, numCols + shiftAmount: numCols] = 0

    elif shiftAmount > 0:
        shifted[:, 0: shiftAmount] = 0

    return shifted