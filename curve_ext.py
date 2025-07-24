import numpy as np

def curve_ext(P, lambda_):
# Extracts a ridge from a TFR using dynamic programming.
# INPUT
#    P          :  TFR with row dimension as time and column dimension as frequency.
#    lambda_    :  Ridge extraction bandwidth.
# OUTPUT
#    c          :  The curve (ridge). 
#    Fval       :  The frequency values.
# Adapted from the MATLAB implementation by Jianfeng Lu into Python by Shrikant Chand (shrikant.chand@duke.edu) and Po-Ying Chen in 04.2024.

    eps = 1e-8
    E = P / np.sum(P)
    E = -np.log(E + eps)
    m, n = E.shape
    FVal = np.full((m, n), np.inf)
    FVal[0, :] = E[0, :]
    c = np.zeros(m, dtype=int)
    for ii in range(1, m):  # time
        for jj in range(n):  # freq
            # Calculate the penalty term
            for kk in range(n):
                FVal[ii, jj] = min(FVal[ii, jj], FVal[ii - 1, kk] + lambda_ * (kk - jj) ** 2)

            # Add the SST value at time ii and freq jj
            FVal[ii, jj] += E[ii, jj]
    c[-1] = np.argmin(FVal[-1, :])
    for ii in range(m - 2, -1, -1):
        val = FVal[ii + 1, c[ii + 1]] - E[ii + 1, c[ii + 1]]
        for kk in range(n):
            if abs(val - FVal[ii, kk] - lambda_ * (kk - c[ii + 1]) ** 2) < eps:
                c[ii] = kk
                break
        if c[ii] == 0:
            c[ii] = n // 2
    return c, FVal
