import numpy as np

def recon_sst(tfrsq, tfrsqtic, Hz, c, Band, coeff):
    alpha = tfrsqtic[1] - tfrsqtic[0]
    RR = round(Band / (Hz * alpha))
    recon = []
    C = 2 * alpha / coeff
    for kk in range(len(c)):
        idx = range(max(0, c[kk] - RR), min(len(tfrsqtic), c[kk] + RR))
        recon.append(C * np.sum(tfrsq[idx, kk]))
    return np.array(recon)
