import numpy as np

def recon_sst(tfrsq, tfrsqtic, Hz, c, Band, coeff):
# Reconstructs asignal using its synchrosqueezed transform's curves.
# INPUT
#    tfrsq     :  Synchrosqueexed TFR.
#    tfrsqtic  :  Frequency axis.
#    Hz        :  Sampling rate (in Hz).
#    c         :  Extracted curve.
#    Band      :  Band width for curve.
#    coeff     :  Frequency resolution coefficient.

# OUTPUT       :  Reconstructed signal
# Adapted from the MATLAB implementation by Hau-Tieng Wu into Python by Shrikant Chand (shrikant.chand@duke.edu) and Po-Ying Chen in 04.2024.
    alpha = tfrsqtic[1] - tfrsqtic[0]
    RR = round(Band / (Hz * alpha))
    recon = []
    C = 2 * alpha / coeff
    for kk in range(len(c)):
        idx = range(max(0, c[kk] - RR), min(len(tfrsqtic), c[kk] + RR))
        recon.append(C * np.sum(tfrsq[idx, kk]))
    return np.array(recon)
